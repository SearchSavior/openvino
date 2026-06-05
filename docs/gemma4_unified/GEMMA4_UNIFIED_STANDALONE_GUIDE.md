# Gemma4 Unified: Standalone Implementation Guide

**Language-agnostic architecture reference with numpy pseudocode and validation against HuggingFace transformers.**

This guide describes the Gemma4 Unified architecture independent of any framework. All logic is expressed in pseudocode/numpy, with complete test suite for validation.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Configuration Schema](#configuration-schema)
3. [Tensor Shapes & Memory Layout](#tensor-shapes--memory-layout)
4. [Key Algorithms](#key-algorithms)
5. [Complete Forward Pass](#complete-forward-pass)
6. [Testing Against Transformers](#testing-against-transformers)

---

## Core Concepts

### Model Type: Decoder-Only Omni-Modal

```
Input: text_ids + image_patches + audio_frames
  ↓
Embedding Layer: Convert to shared (batch, seq_len, hidden_size) tensors
  ↓
Stack of N Transformer Blocks:
  ├─ Hybrid Attention (sliding-window or full-context)
  ├─ Per-layer RoPE (different for sliding vs full)
  ├─ Residual connections
  ├─ LayerNorm (RMSNorm variant)
  ├─ FFN (feed-forward network)
  └─ Optional: per-layer scalar multiplier
  ↓
Final RMSNorm + Linear projection
  ↓
Output: logits (batch, seq_len, vocab_size)
```

### Key Innovation: Encoder-Free

Traditional VLMs:
```
Image → ViT encoder → frozen embeddings → merge with text
Audio → Conformer → frozen embeddings → merge with text
```

Gemma4 Unified:
```
Image patches → lightweight projection → text embedding space (unified)
Audio frames  → lightweight projection → text embedding space (unified)
```

---

## Configuration Schema

### Top-Level Config

```python
class Gemma4UnifiedConfig:
    # Sub-configs
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: AudioConfig
    
    # Special token IDs
    boi_token_id: int = 255999     # begin-of-image
    eoi_token_id: int = 258882     # end-of-image
    image_token_id: int = 258880   # image placeholder
    boa_token_id: int = 256000     # begin-of-audio
    eoa_token_id: int = 258883     # end-of-audio
    audio_token_id: int = 258881   # audio placeholder
    video_token_id: int = 258884   # video placeholder
```

### Text Config

```python
class TextConfig:
    # Dimensions
    vocab_size: int = 262144           # 262K token vocabulary
    hidden_size: int = 3840            # hidden dimension
    num_hidden_layers: int = 48        # 48 decoder layers
    num_attention_heads: int = 16      # Q heads
    num_key_value_heads: int = 8       # KV heads (group-query attention)
    head_dim: int = 256                # per-head dimension (sliding layers)
    global_head_dim: int = 512         # per-head dimension (full layers)
    intermediate_size: int = 9216      # FFN hidden (4× hidden_size)
    max_position_embeddings: int = 262144  # max context length
    sliding_window: int = 1024         # sliding attention window
    
    # Features
    use_cache: bool = True             # enable KV caching
    use_bidirectional_attention: str = "vision"  # "vision" or "all"
    attention_k_eq_v: bool = False     # K=V projection sharing
    num_kv_shared_layers: int = 0      # consecutive layers sharing KV
    
    # Activation & norm
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6         # RMSNorm epsilon
    
    # Token IDs
    pad_token_id: int = 0
    eos_token_id: int | list[int] = 1
    bos_token_id: int = 2
    
    # Layer types & RoPE
    layer_types: list[str]             # "sliding_attention" or "full_attention"
    rope_parameters: dict              # per-layer-type RoPE config
    
    final_logit_softcapping: float | None = None  # tanh softcapping scale
```

### Vision Config

```python
class VisionConfig:
    patch_size: int = 16               # raw image patch size (pixels)
    pooling_kernel_size: int = 3       # merge kernel (3×3)
    model_patch_size: int = 48         # merged patch size (16 * 3)
    mm_embed_dim: int = 3840           # multimodal hidden
    mm_posemb_size: int = 1120         # positional embedding table size
    output_proj_dims: int = 3840       # projection output (= text hidden)
    rms_norm_eps: float = 1e-6
```

### Audio Config

```python
class AudioConfig:
    audio_embed_dim: int = 640         # samples per token (40ms @ 16kHz)
    audio_samples_per_token: int = 640 # same
    rms_norm_eps: float = 1e-6
```

---

## Tensor Shapes & Memory Layout

### Input Tensors

**Text:**
```
input_ids: (batch_size, seq_len) dtype=int32
  seq_len typically 512-262144
  values in range [0, vocab_size)
```

**Image:**
```
pixel_values: (batch_size, max_patches, patch_dim)
  max_patches = 280 (typical, or 70/140/560/1120)
  patch_dim = 6912 (48×48×3 pixels)
  dtype = float32 initially, cast to bfloat16 in processing

image_position_ids: (batch_size, max_patches, 2)
  values are (x, y) coordinates in image grid
  invalid patches marked as (-1, -1)
```

**Audio:**
```
input_features: (batch_size, num_frames, frame_size)
  frame_size = 640 (raw waveform samples)
  num_frames varies per audio
  dtype = float32

input_features_mask: (batch_size, num_frames) dtype=bool
  True = valid frame, False = padding
```

### Intermediate Tensors

**After embedding:**
```
embeddings: (batch_size, seq_len, hidden_size)
  seq_len = text_len + image_tokens + audio_tokens + video_tokens
  hidden_size = 3840 for 12B model
```

**After each attention block:**
```
hidden_states: (batch_size, seq_len, hidden_size)
  shape preserved through residual connections
```

**Query, Key, Value in attention:**
```
query: (batch_size, seq_len, num_heads, head_dim)
key: (batch_size, seq_len, num_kv_heads, head_dim)
value: (batch_size, seq_len, num_kv_heads, head_dim)
```

**Output logits:**
```
logits: (batch_size, seq_len, vocab_size)
  shape[2] = 262144 for full model, 512 for tiny model
```

---

## Key Algorithms

### 1. RMSNorm (Root Mean Square Layer Normalization)

```python
def rms_norm(x: ndarray, weight: ndarray, eps: float = 1e-6) -> ndarray:
    """
    RMSNorm: normalize by RMS, then scale by weight.
    
    Args:
        x: (..., d) input tensor
        weight: (d,) learnable scale
        eps: numerical stability epsilon
    
    Returns:
        (..., d) normalized tensor
    """
    # Compute RMS along last dimension
    rms = sqrt(mean(x**2, axis=-1, keepdims=True) + eps)
    
    # Normalize
    x_norm = x / rms
    
    # Scale
    return x_norm * weight
```

### 2. Rotary Position Embeddings (RoPE)

```python
def rope_forward(
    x: ndarray,           # (batch, seq_len, heads, head_dim)
    position_ids: ndarray,  # (batch, seq_len)
    inv_freq: ndarray,    # (head_dim // 2,)
) -> tuple[ndarray, ndarray]:
    """
    Apply rotary position embeddings.
    
    Returns:
        cos: (..., head_dim)
        sin: (..., head_dim)
    """
    # Compute angles: position * inv_freq
    # inv_freq = 1 / (theta ** (2i / d))
    inv_freq_expanded = inv_freq[None, :, None]  # (1, d//2, 1)
    position_ids_expanded = position_ids[:, None, :]  # (batch, 1, seq_len)
    
    # freqs: (batch, d//2, seq_len)
    freqs = einsum('bij,jk->bik', inv_freq_expanded, position_ids_expanded)
    freqs = freqs.transpose(0, 2, 1)  # (batch, seq_len, d//2)
    
    # Duplicate to full dimension
    emb = concatenate([freqs, freqs], axis=-1)  # (batch, seq_len, d)
    
    # Compute cos and sin
    cos_vals = cos(emb)
    sin_vals = sin(emb)
    
    return cos_vals, sin_vals


def apply_rope(x: ndarray, cos: ndarray, sin: ndarray) -> ndarray:
    """
    Apply rotation to x using cos/sin embeddings.
    
    x shape: (..., d)
    cos/sin shape: (..., d)
    """
    # Split x into two halves
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    
    # Rotate
    # [x1, x2] * rotation = [x1*cos - x2*sin, x1*sin + x2*cos]
    cos_half = cos[..., :cos.shape[-1]//2]
    sin_half = sin[..., :sin.shape[-1]//2]
    
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half
    
    return concatenate([rotated_x1, rotated_x2], axis=-1)
```

### 3. Patches Merge (Vision Pipeline)

```python
def patches_merge(
    patches: ndarray,      # (batch, L, 768) teacher patches
    positions_xy: ndarray, # (batch, L, 2) XY coordinates
    target_length: int,    # target num patches (L / 9 for 3×3)
) -> tuple[ndarray, ndarray]:
    """
    Merge k×k spatially-adjacent patches, preserving locality.
    
    Algorithm:
    1. Compute spatial grouping (x//k, y//k)
    2. Compute target ordering to group patches into k×k kernels
    3. Permute patches into kernel order via argsort
    4. Reshape into merged patches (target_length, k²×D)
    5. Update positions (take min per kernel, divide by k)
    """
    patch_size = int(sqrt(patches.shape[-1] / 3))  # 16
    k = int(sqrt(patches.shape[1] / target_length))  # 3
    
    # Step 1: Spatial grouping
    kernel_idxs = floor(positions_xy / k)  # (batch, L, 2)
    
    # Step 2: Compute target ordering
    max_x = max(positions_xy[..., 0]) + 1
    num_patches_from_top_left = (
        k * k * kernel_idxs[..., 0] +
        k * max_x * kernel_idxs[..., 1]
    )
    
    position_within_kernel = mod(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[..., 0] +
        position_within_kernel[..., 1] * k
    )
    target_ordering = (
        num_patches_from_top_left_of_kernel +
        num_patches_from_top_left
    )
    
    # Step 3: Permute via argsort
    perm = argsort(target_ordering.long(), axis=1)  # inverse permutation
    # Gather patches by permutation
    kernel_ordered_patches = gather(patches, perm)  # (batch, L, 768)
    
    # Step 4: Reshape
    # (batch, L, 768) → (batch, target_length, 9, 16, 16, 3)
    batch_size = patches.shape[0]
    kernel_ordered_patches = reshape(
        kernel_ordered_patches,
        (batch_size, target_length, k*k, patch_size, patch_size, 3)
    )
    # Rearrange spatial dimensions
    kernel_ordered_patches = reshape(
        kernel_ordered_patches,
        (batch_size, target_length, k, k, patch_size, patch_size, 3)
    )
    # Permute: (batch, target_length, k, patch_size, k, patch_size, 3)
    kernel_ordered_patches = transpose(
        kernel_ordered_patches,
        (0, 1, 2, 4, 3, 5, 6)
    )
    merged_patches = reshape(
        kernel_ordered_patches,
        (batch_size, target_length, k*patch_size, k*patch_size, 3)
    )
    # Flatten last 3 dims: (batch, target_length, 6912)
    merged_patches = reshape(
        merged_patches,
        (batch_size, target_length, -1)
    )
    
    # Step 5: Update positions
    kernel_ordered_positions = gather(positions_xy, perm)
    kernel_ordered_positions = reshape(
        kernel_ordered_positions,
        (batch_size, target_length, k*k, 2)
    )
    new_positions = floor(kernel_ordered_positions.mean(axis=2) / k)
    
    return merged_patches, new_positions.long()
```

### 4. Attention with Sliding Window

```python
def sliding_window_attention(
    query: ndarray,       # (batch, seq_len, heads, head_dim)
    key: ndarray,         # (batch, seq_len, kv_heads, head_dim)
    value: ndarray,       # (batch, seq_len, kv_heads, head_dim)
    attention_mask: ndarray,  # (batch, 1, seq_len, seq_len) or None
    window_size: int = 1024,
    dropout_p: float = 0.0,
) -> tuple[ndarray, ndarray]:
    """
    Compute attention with sliding window.
    
    For each position i, attend to [max(0, i - window_size), i] (causal)
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # Repeat KV to match number of query heads (group-query attention)
    key = repeat(key, seq_len, num_heads)  # (batch, seq_len, num_heads, head_dim)
    value = repeat(value, seq_len, num_heads)
    
    # Compute attention scores
    scores = einsum('bshi,bshi->bhij', query, key)  # (batch, heads, seq_len, seq_len)
    scores = scores / sqrt(head_dim)
    
    # Apply sliding window mask
    # Create causal mask with sliding window
    window_mask = zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - window_size)
        window_mask[i, start:i+1] = 1.0
        window_mask[i, i+1:] = 0.0  # Future masked (causal)
    
    # Convert to attention bias
    window_bias = (1 - window_mask) * -1e9
    scores = scores + window_bias[None, None, ...]
    
    # Apply attention mask if provided
    if attention_mask is not None:
        scores = scores + attention_mask
    
    # Softmax
    attn_weights = softmax(scores, axis=-1)
    attn_weights = dropout(attn_weights, p=dropout_p)
    
    # Apply to values
    output = einsum('bhij,bshd->bshd', attn_weights, value)
    
    return output, attn_weights
```

---

## Complete Forward Pass

```python
def forward_pass(
    model_config: Gemma4UnifiedConfig,
    input_ids: ndarray,                # (batch, seq_len)
    pixel_values: ndarray | None = None,     # (batch, max_patches, 6912)
    image_position_ids: ndarray | None = None,
    input_features: ndarray | None = None,   # (batch, num_frames, 640)
    input_features_mask: ndarray | None = None,
    position_ids: ndarray | None = None,     # (batch, seq_len)
    attention_mask: ndarray | None = None,
    block_sequence_ids: ndarray | None = None,  # for bidirectional vision
) -> ndarray:
    """
    Full forward pass through Gemma4 Unified.
    
    Returns:
        logits: (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    text_config = model_config.text_config
    
    # ===== STEP 1: Embed text =====
    embeddings = embedding_layer(input_ids)  # (batch, seq_len, hidden)
    
    # ===== STEP 2: Embed and scatter vision =====
    if pixel_values is not None:
        # Vision pipeline
        x_vision = vision_embedder(pixel_values, image_position_ids)
        # x_vision: (total_valid_patches, hidden)
        
        # Scatter into embeddings (via masked_scatter)
        image_mask = extract_image_tokens(input_ids, text_config.image_token_id)
        embeddings = scatter_into_embeddings(embeddings, x_vision, image_mask)
    
    # ===== STEP 3: Embed and scatter audio =====
    if input_features is not None and input_features_mask is not None:
        # Audio pipeline
        x_audio = audio_embedder(input_features)  # (batch, num_frames, hidden)
        
        # Strip padding
        x_audio = x_audio[input_features_mask]  # (valid_tokens, hidden)
        
        # Scatter into embeddings
        audio_mask = extract_audio_tokens(input_ids, text_config.audio_token_id)
        embeddings = scatter_into_embeddings(embeddings, x_audio, audio_mask)
    
    # ===== STEP 4: Prepare position IDs and masks =====
    if position_ids is None:
        position_ids = arange(seq_len).unsqueeze(0)  # (1, seq_len)
    
    if attention_mask is None and block_sequence_ids is not None:
        # Build bidirectional vision mask
        attention_mask = build_bidirectional_vision_mask(
            seq_len, block_sequence_ids, text_config
        )
    elif attention_mask is None:
        # Standard causal mask
        attention_mask = build_causal_mask(seq_len)
    
    # ===== STEP 5: Transformer blocks =====
    hidden_states = embeddings
    
    for layer_idx in range(text_config.num_hidden_layers):
        layer_type = text_config.layer_types[layer_idx]
        rope_params = text_config.rope_parameters[layer_type]
        
        # RoPE embeddings for this layer
        cos, sin = rope_init(rope_params, position_ids, text_config)
        
        # Attention
        query = linear_q(hidden_states)  # (batch, seq_len, hidden)
        query = reshape(query, (batch_size, seq_len, text_config.num_attention_heads, -1))
        
        key = linear_k(hidden_states)
        key = reshape(key, (batch_size, seq_len, text_config.num_key_value_heads, -1))
        
        value = linear_v(hidden_states)
        value = reshape(value, (batch_size, seq_len, text_config.num_key_value_heads, -1))
        
        # Apply RoPE
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        
        # Compute attention
        if layer_type == "sliding_attention":
            attn_output, _ = sliding_window_attention(
                query, key, value, attention_mask,
                window_size=text_config.sliding_window
            )
        else:  # full_attention
            attn_output, _ = full_attention(
                query, key, value, attention_mask
            )
        
        attn_output = reshape(attn_output, (batch_size, seq_len, -1))
        attn_output = linear_o(attn_output)
        
        # Residual
        hidden_states = hidden_states + attn_output
        
        # FFN
        ffn_input = rms_norm(hidden_states)
        gate = linear_gate(ffn_input)
        up = linear_up(ffn_input)
        ffn_out = gate * gelu(up)
        ffn_out = linear_down(ffn_out)
        
        # Residual + layer scalar
        hidden_states = hidden_states + ffn_out * layer_scalar[layer_idx]
    
    # ===== STEP 6: Final norm and output projection =====
    hidden_states = rms_norm(hidden_states)
    logits = linear_lm_head(hidden_states)  # (batch, seq_len, vocab_size)
    
    return logits
```

---

## Testing Against Transformers

See `test_gemma4_unified.py` for complete validation suite.

**Key test strategy:**
1. Load tiny model with transformers
2. Create identical dummy inputs
3. Run forward pass on both numpy and transformers
4. Compare logits (should match to float precision)
5. Verify all intermediate shapes

**Test inputs:**
```python
batch_size = 2
seq_len = 128
vocab_size = 512
hidden_size = 128

# Dummy inputs
input_ids = randint(0, vocab_size, (batch_size, seq_len))
position_ids = arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# Optional: vision
pixel_values = randn(batch_size, 70, 6912)
image_position_ids = arange(70).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 2)

# Optional: audio
input_features = randn(batch_size, 256, 640)
input_features_mask = ones(batch_size, 256, dtype=bool)
```

**Expected behavior:**
- Logits shape: (batch_size, seq_len, vocab_size)
- Logits dtype: bfloat16
- Values in reasonable range (~[-10, 10])

---

## Implementation Notes

### Memory Layout
- All tensors stored as float32 during computation, cast to bfloat16 for storage
- Attention matrices: (batch, heads, seq_len, seq_len) — can be large; consider chunking
- KV cache: (batch, seq_len, num_kv_heads, head_dim) — enables fast inference

### Precision
- bfloat16: 16-bit float, good for training and inference
- Conversion: explicit casting before matmul operations
- RMSNorm epsilon: 1e-6 (prevents numerical issues)

### Optimization Opportunities
- Flash Attention for faster attention (O(N) vs O(N²) memory)
- KV caching during generation
- Quantization (int8, int4)
- Batching across modalities

---

**Next**: See `test_gemma4_unified.py` for complete test suite with validation against transformers.

