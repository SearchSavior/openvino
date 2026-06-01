"""Python-level IR rewrites that reduce the number of [1, T, X] f32
intermediates in each linear_attn layer of Qwen3.5-VL.

All rewrites use plugin-native ops only -- nothing custom. They stay
inside the supported API and keep the plugin's MemMgr able to pool
through the new ops.

Findings live in DISCUSSION.md, not here."""
import numpy as np
import openvino as ov
from openvino import opset15 as ops


def _find_qkv_split_chain(model):
    """Yield (split_node, source_3d, three_reshapes) for every VariadicSplit
    on a [?, ?, 6144] input whose three outputs feed three Reshapes to
    [?, ?, 16, 128] within a linear_attn module."""
    for op in model.get_ops():
        if op.get_type_name() != "VariadicSplit":
            continue
        name = op.get_friendly_name()
        if "linear_attn" not in name:
            continue
        src = op.input(0).get_source_output()
        if src.get_partial_shape().rank.get_length() != 3:
            continue
        if not src.get_partial_shape()[2].is_static or src.get_partial_shape()[2].get_length() != 6144:
            continue
        if op.get_output_size() != 3:
            continue
        reshapes = []
        ok = True
        for i in range(3):
            consumers = list(op.output(i).get_target_inputs())
            # All consumers of this split output must be Reshapes (or all the
            # same Reshape). PyTorch emits exactly one Reshape per split output.
            rs = None
            for c in consumers:
                cn = c.get_node()
                if cn.get_type_name() != "Reshape":
                    ok = False; break
                ps = cn.get_output_partial_shape(0)
                if ps.rank.get_length() != 4 \
                        or not ps[2].is_static or ps[2].get_length() != 16 \
                        or not ps[3].is_static or ps[3].get_length() != 128:
                    ok = False; break
                if rs is None: rs = cn
                elif rs is not cn:
                    # multiple distinct Reshapes per split output -- skip
                    ok = False; break
            if not ok or rs is None:
                ok = False; break
            reshapes.append(rs)
        if not ok:
            continue
        yield op, src, reshapes


def reshape_before_split(model: ov.Model) -> int:
    """Replace the eager-style 'VariadicSplit on dim 2 (channel) then
    three Reshapes to [B, T, 16, 128]' pattern with the equivalent
    'Reshape to [B, T, 48, 128] then VariadicSplit on dim 2 (heads)'.
    Eliminates three [?, T, 2048] f32 intermediates per linear-attn layer.

    Returns the number of patterns rewritten."""
    rewrites = list(_find_qkv_split_chain(model))
    n = 0
    for split, src, reshapes in rewrites:
        # New: pre-Reshape the [?, ?, 6144] source to [?, ?, 48, 128].
        # Reshape with target [0, 0, 48, 128]: the two zeros copy the
        # first two dims from the input; 48 = 16 (heads per stream) * 3 streams.
        pre_shape = ops.constant(np.array([0, 0, 48, 128], dtype=np.int64))
        pre_reshape = ops.reshape(src, pre_shape, special_zero=True)
        pre_reshape.set_friendly_name(split.get_friendly_name() + "/PreReshape")

        # New VariadicSplit on axis -2 (head dim), three groups of 16 heads.
        axis_const = ops.constant(np.array([-2], dtype=np.int64))
        sizes_const = ops.constant(np.array([16, 16, 16], dtype=np.int64))
        new_split = ops.variadic_split(pre_reshape, axis_const, sizes_const)
        new_split.set_friendly_name(split.get_friendly_name() + "/PreSplit")

        # Rewire: every consumer of each Reshape's output should consume the
        # corresponding new_split output instead. Reshape itself becomes dead.
        for i, rs in enumerate(reshapes):
            rs.output(0).replace(new_split.output(i))
        n += 1
    return n


def fold_q_scale_into_rsqrt(model: ov.Model) -> int:
    """In the Q path of each linear_attn layer, fold the final
        Divide(transposed_q, Power(D, 0.5))
    into the rsqrt by multiplying rsqrt by 1/sqrt(D) before the Multiply.

    Pattern detection:
        Divide(_, Power(_, 0.5))  with friendly_name containing 'linear_attn'
        and the Divide's input(0) being a Transpose of a Multiply whose
        second input is a Divide (the rsqrt path).

    The rewrite avoids a [?, 16, ?, 128] f32 intermediate per layer.

    Returns the number of patterns rewritten."""
    n = 0
    for op in list(model.get_ops()):
        if op.get_type_name() != "Divide":
            continue
        name = op.get_friendly_name()
        if "linear_attn" not in name:
            continue
        # in[1] should be a Power node returning scalar shape []
        in1 = op.input(1).get_source_output().get_node()
        if in1.get_type_name() != "Power":
            continue
        if in1.get_output_partial_shape(0).rank.get_length() != 0:
            continue
        # in[0] should be a Transpose of a Multiply
        in0_node = op.input(0).get_source_output().get_node()
        if in0_node.get_type_name() != "Transpose":
            continue
        mul_q = in0_node.input(0).get_source_output().get_node()
        if mul_q.get_type_name() != "Multiply":
            continue
        # The "rsqrt-ish" branch of that Multiply should be a Divide.
        rsqrt_branch = None; data_branch = None
        for k in range(mul_q.get_input_size()):
            src = mul_q.input(k).get_source_output()
            if src.get_node().get_type_name() == "Divide":
                rsqrt_branch = src
            else:
                data_branch = src
        if rsqrt_branch is None or data_branch is None:
            continue

        # PyTorch exports q.shape[-1] as ShapeOf(MULTIPLY_q) -> Gather -> Power.
        # If we make MULTIPLY_q's rsqrt branch depend on that Power, we create
        # MULTIPLY_q -> ShapeOf -> ... -> Power -> rsqrt -> MULTIPLY_q (cycle).
        # head_dim is statically 128 in this architecture, so just use a const.
        head_dim = mul_q.get_output_partial_shape(0)[-1]
        if not head_dim.is_static:
            continue
        inv_sqrt_d_val = np.float32(1.0 / np.sqrt(head_dim.get_length()))
        inv_sqrt_d = ops.constant(inv_sqrt_d_val)
        inv_sqrt_d.set_friendly_name(name + "/InvSqrtD")

        scaled_rsqrt = ops.multiply(rsqrt_branch, inv_sqrt_d)
        scaled_rsqrt.set_friendly_name(name + "/RsqrtScaled")

        # Wire the Multiply that produced normalized Q to use scaled_rsqrt.
        # We need to find the input index that matched rsqrt_branch.
        for k in range(mul_q.get_input_size()):
            if mul_q.input(k).get_source_output() == rsqrt_branch:
                mul_q.input(k).replace_source_output(scaled_rsqrt.output(0))
                break

        # Now the Divide(by sqrt(D)) is redundant: its result equals its input.
        # Bypass it by rewiring every consumer to consume the Transpose output.
        op.output(0).replace(in0_node.output(0))
        n += 1
    return n


def fuse_l2_norm(model: ov.Model) -> int:
    """Pattern: Multiply(x, x) -> ReduceSum(axes) -> Add(eps) -> Sqrt -> Divide(1, Sqrt) -> Multiply(x, _)
    Replace with NormalizeL2(x, axes, eps, ADD). Collapses 6 ops into 1.

    Returns the number of patterns rewritten."""
    n = 0
    for op in list(model.get_ops()):
        if op.get_type_name() != "Multiply":
            continue
        if "linear_attn" not in op.get_friendly_name():
            continue
        # Find the "Multiply(x, x)" square. Op is the *outer* normalizing
        # multiply (Multiply(x, 1/rsqrt)). Walk: in[1] = Divide, in[0] = x.
        in1 = op.input(1).get_source_output().get_node()
        if in1.get_type_name() != "Divide":
            continue
        # Divide(1, Sqrt(...))
        sqrt_node = in1.input(1).get_source_output().get_node()
        if sqrt_node.get_type_name() != "Sqrt":
            continue
        add_node = sqrt_node.input(0).get_source_output().get_node()
        if add_node.get_type_name() != "Add":
            continue
        # Add(ReduceSum, eps_const)
        reduce_node = None; eps_const = None
        for k in range(2):
            src = add_node.input(k).get_source_output().get_node()
            if src.get_type_name() == "ReduceSum":
                reduce_node = src
            elif src.get_type_name() == "Constant":
                eps_const = src
        if reduce_node is None or eps_const is None:
            continue
        # ReduceSum(Multiply(x, x), axes_const)
        square_mul = reduce_node.input(0).get_source_output().get_node()
        if square_mul.get_type_name() != "Multiply":
            continue
        # Confirm Multiply is squaring (both inputs same).
        if square_mul.input(0).get_source_output() != square_mul.input(1).get_source_output():
            continue
        x_src = square_mul.input(0).get_source_output()
        # The outer Multiply (`op`) should take `x` as its first input.
        if op.input(0).get_source_output() != x_src:
            continue
        axes_src = reduce_node.input(1).get_source_output()

        # eps value
        eps_val = float(eps_const.get_data().flatten()[0])

        norm = ops.normalize_l2(x_src, axes_src, eps_val, "add")
        norm.set_friendly_name(op.get_friendly_name() + "/NormL2")
        op.output(0).replace(norm.output(0))
        n += 1
    return n


def reduce_linear_attn_intermediates(model: ov.Model) -> dict:
    """Apply all plugin-native IR reductions for linear-attn modules.
    Returns counts of patterns rewritten per pass."""
    return {
        "reshape_before_split": reshape_before_split(model),
        "fold_q_scale_into_rsqrt": fold_q_scale_into_rsqrt(model),
        "fuse_l2_norm": fuse_l2_norm(model),
    }
