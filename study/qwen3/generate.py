"""Generate from Qwen3.5-VL LM with all three fusions applied, greedy decode."""
import sys, time
sys.path.insert(0, "/home/user/openvino/study/qwen3")
import numpy as np
import openvino as ov
from transformers import AutoTokenizer
from fused_linear_attn import register as register_la, replace_gated_delta_rule_loops
from fused_conv1d import register as register_cv, replace_causal_conv1d_chains
from lm_head_slice import slice_lm_head_to_last_token

MODEL_DIR = "/tmp/qwen3-work/qwen35-0.8b-int8"
PROMPT = "What is your show size?"
MAX_NEW_TOKENS = 64
APPLY_FUSIONS = True


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"=== templated prompt ===\n{prompt_text!r}\n=== end ===")
    input_ids = np.asarray([tok.encode(prompt_text)], dtype=np.int64)
    print(f"input_ids shape: {input_ids.shape}")

    core = ov.Core()
    if APPLY_FUSIONS:
        register_la(core)
        register_cv(core)

    embed_model = core.compile_model(f"{MODEL_DIR}/openvino_text_embeddings_model.xml", "CPU")
    lm = core.read_model(f"{MODEL_DIR}/openvino_language_model.xml")
    if APPLY_FUSIONS:
        print(f"linear-attn replaced: {replace_gated_delta_rule_loops(lm)}")
        print(f"conv1d replaced:      {replace_causal_conv1d_chains(lm)}")
        print(f"lm_head slice:        {slice_lm_head_to_last_token(lm)}")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})

    def embed(ids):
        return list(embed_model.create_infer_request().infer({0: ids}).values())[0]

    req = compiled.create_infer_request()
    logits_out = next(o for o in compiled.outputs if "logits" in o.get_any_name())
    T = input_ids.shape[1]

    # Prefill.
    t0 = time.time()
    prefill_embeds = embed(input_ids)
    req.infer({
        "inputs_embeds": prefill_embeds,
        "attention_mask": np.ones((1, T), dtype=np.int64),
        "position_ids": np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    })
    t_prefill = time.time() - t0
    logits = np.asarray(req.get_tensor(logits_out).data)
    next_id = int(logits[0, -1].argmax())
    print(f"\nPrefill: {t_prefill:.2f}s ({T} tokens)")
    print(f"First generated token: {next_id} = {tok.decode([next_id])!r}")

    eos_ids = {tok.eos_token_id}
    if "<|im_end|>" in tok.get_vocab():
        eos_ids.add(tok.convert_tokens_to_ids("<|im_end|>"))

    generated = [next_id]
    past_len = T
    t1 = time.time()
    for step in range(MAX_NEW_TOKENS - 1):
        ne = embed(np.array([[next_id]], dtype=np.int64))
        req.infer({
            "inputs_embeds": ne,
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.full((4, 1, 1), past_len, dtype=np.int64),
            "beam_idx": np.zeros((1,), dtype=np.int32),
        })
        next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())
        generated.append(next_id)
        past_len += 1
        if next_id in eos_ids:
            break
    t_decode = time.time() - t1
    text = tok.decode(generated, skip_special_tokens=False)
    text_clean = tok.decode(generated, skip_special_tokens=True)

    print(f"Decode:  {t_decode:.2f}s ({len(generated)} tokens, {len(generated)/t_decode:.2f} tok/s)")
    print(f"\n=== generated (with specials) ===\n{text}")
    print(f"\n=== generated (clean) ===\n{text_clean}")


if __name__ == "__main__":
    main()
