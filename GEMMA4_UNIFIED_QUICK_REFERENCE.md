# Gemma4 Unified: Quick Reference Card

## Model Specs at a Glance

| Property | Value |
|----------|-------|
| **Architecture** | Decoder-only, omni-modal (text + image + video + audio) |
| **Modality Support** | 4 (text, vision, audio, video) |
| **Context Length** | 262,144 tokens |
| **Vocab Size** | 262,144 (SentencePiece) |
| **Hidden Size** | 3,840 |
| **Num Layers** | 48 (12B) or 30 (smaller variants) |
| **Num Attention Heads** | 16 (8 KV heads, 1 global KV for full attention) |
| **Head Dimension** | 256 (sliding), 512 (full attention) |
| **Attention Pattern** | 5:1 sliding-window + full (hybrid) |
| **Sliding Window Size** | 1,024 tokens |
| **RoPE Base (Sliding)** | θ = 10,000 |
| **RoPE Base (Full)** | θ = 1,000,000 (proportional with 0.25 rotary) |
| **FFN Hidden** | 9,216 (2.4× expansion) |
| **Vision Patch Size** | 16 px → 48 px (after 3×3 merge) |
| **Vision Embed Dim** | 3,840 |
| **Vision Positional Embeddings** | Factorized 2D, 1,120-entry lookup table |
| **Audio Frame Size** | 640 samples (40 ms @ 16 kHz) |
| **Key Innovation** | **Encoder-free**: No vision/audio towers; direct projection |

---

## Input Specifications

### Text
- **Type**: String(s)
- **Tokenizer**: SentencePiece (262,144 vocab)
- **Special tokens**: `<bos>` (2), `<eos>` (1), `<pad>` (0)

### Image
- **Format**: PIL.Image, numpy (H, W, 3), or torch (3, H, W)
- **Preprocessing**: Aspect-ratio-preserving resize, optional rescale/normalize
- **Patch Budget**: `max_soft_tokens` ∈ {70, 140, 280, 560, 1,120}
- **Output per image**: (max_soft_tokens, 6,912) patches → (actual_tokens, 3,840) embeddings
- **Position tracking**: (max_soft_tokens, 2) XY coordinates per patch

### Video
- **Format**: numpy (num_frames, H, W, 3) or torch (num_frames, H, W, 3)
- **Frame sampling**: 32 frames default
- **Patch budget**: Lower than images (e.g., 70 tokens/frame)
- **Output**: (batch, num_frames, max_soft_tokens, 3,840)

### Audio
- **Format**: numpy array (num_samples,) @ 16 kHz
- **Feature extraction**: Raw waveform chunked to 640-sample frames (no mel-spec)
- **Output**: (batch, num_frames, 640) → (num_frames, 3,840) via projection

---

## Key Architectural Patterns

### 1. Hybrid Attention Scheduling

```
Layer pattern (every 6 layers, default):
[sliding, sliding, sliding, sliding, sliding, full, ...]

Per-layer RoPE:
- Sliding: θ=10k, head_dim=256
- Full: θ=1M, head_dim=512, proportional rotation (0.25 of dims)
```

### 2. Bidirectional Vision, Causal Text

```
Vision tokens (image_token_id, video_token_id):
  ├─ Attend to all other vision tokens (bidirectional)
  └─ Attend to all text tokens

Text tokens:
  ├─ Attend to all previous tokens (causal)
  └─ Cannot see future text tokens
```

### 3. Encoder-Free Modality Pipeline

```
Image/Video:
  patches → merge (3×3 spatial pooling) → dense proj
  → +factorized 2D posemb → multimodal proj

Audio:
  raw frames → multimodal proj (direct, no conformer)

Result: All modalities in shared (batch, seq, 3840) embedding space
```

### 4. KV Sharing (Optional)

```
Last N layers reuse K,V from earlier layers:
  Layer 44-47 reuse from layers 40-43 (when num_kv_shared_layers=4)
```

### 5. Per-Layer Learnable Scalar

```
hidden_states *= layer_scalar  (initialized to 1.0, trained)
```

---

## Special Tokens

| Token | ID | Purpose |
|-------|----|---------| 
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 1 | End of sequence |
| `<pad>` | 0 | Padding |
| `<image>` | 258,880 | Image placeholder |
| `<video>` | 258,884 | Video placeholder |
| `<audio>` | 258,881 | Audio placeholder |
| `<boi>` | 255,999 | Begin-of-image wrapper |
| `<eoi>` | 258,882 | End-of-image wrapper |
| `<boa>` | 256,000 | Begin-of-audio wrapper |
| `<eoa>` | 258,883 | End-of-audio wrapper |

---

## Processing Pipeline

```
Input (text + images + audio)
  ↓
Tokenizer (text) + ImageProcessor (images) + FeatureExtractor (audio)
  ↓
Placeholder identification & dynamic token budget
  ↓
Modality-specific embedders:
  - Vision: patch_ln1 → dense → patch_ln2 → +posemb → multimodal_proj
  - Audio: multimodal_proj (direct)
  ↓
Scatter into text embeddings via masked_scatter
  ↓
Shared transformer (48 layers, hybrid attention)
  ↓
RMSNorm → LM head (3840 → 262144)
  ↓
Logits → Loss/Generation
```

---

## Important Configuration Attributes

### `Gemma4UnifiedTextConfig`

```python
layer_types: list[str]  # ["sliding_attention"] * 5 + ["full_attention"]
use_bidirectional_attention: str  # "vision" or "all"
sliding_window: int = 1024
num_global_key_value_heads: int | None = None  # for full attention
global_head_dim: int = 512
attention_k_eq_v: bool = False
num_kv_shared_layers: int = 0
use_double_wide_mlp: bool = False
final_logit_softcapping: float | None = None
```

### `Gemma4UnifiedVisionConfig`

```python
patch_size: int = 16
pooling_kernel_size: int = 3
model_patch_size: int = 48  # (computed: patch_size * pooling_kernel_size)
mm_embed_dim: int = 3840
mm_posemb_size: int = 1120
```

### `Gemma4UnifiedAudioConfig`

```python
audio_embed_dim: int = 640
audio_samples_per_token: int = 640  # (same as above)
```

---

## Inference Checklist

- [ ] Load model with `torch_dtype=torch.bfloat16` (or float16)
- [ ] Use `device_map="auto"` for automatic VRAM optimization
- [ ] Preprocess images with `max_soft_tokens` (280 typical)
- [ ] Audio must be 16 kHz resampled
- [ ] Use `model.generate()` with:
  - `max_new_tokens`: tokens to generate
  - `temperature`: 0.7-0.9 for sampling
  - `top_p`: 0.9-0.95 for nucleus sampling
  - `do_sample=True` for non-greedy
- [ ] Tokenizer decoding: `processor.tokenizer.decode(output_ids[0])`
- [ ] For streaming: use `TextIteratorStreamer`
- [ ] Batch inference: nest image lists for multiple images

---

## Training Checklist

- [ ] Use LoRA for parameter-efficient fine-tuning (r=16, alpha=32)
- [ ] Target modules: `q_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`
- [ ] Learning rate: 2e-5 (for LoRA) or 1e-5 (full)
- [ ] Warmup steps: ~500 for 10k-step training
- [ ] Gradient checkpointing: `gradient_checkpointing_enable()`
- [ ] Mixed precision: `torch.bfloat16` for stability
- [ ] Distributed training: Use `DistributedDataParallel`
- [ ] Loss computation: `outputs.loss` on labels-provided forward pass
- [ ] Optimizer: AdamW with weight_decay=0.01

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA OOM | Batch size too large | ↓ batch_size, use gradient_checkpointing |
| Slow inference | No KV cache | Ensure `use_cache=True` in generate() |
| Poor image understanding | Too few soft tokens | Increase `max_soft_tokens` (280→560) |
| Audio not recognized | Wrong sampling rate | Resample to 16 kHz |
| Mismatch: image tokens vs features | Padding/slicing mismatch | Check `num_soft_tokens_per_image` |
| Vision tokens see future text | Wrong attention mask | Use `use_bidirectional_attention="vision"` |

---

## Performance Metrics (Approximate)

| Metric | 12B Model |
|--------|-----------|
| **Inference latency** (single image) | ~200-300 ms (A100) |
| **Throughput** (batch=4) | ~30-50 tokens/sec |
| **VRAM (fp32)** | ~24 GB |
| **VRAM (bfloat16)** | ~12 GB |
| **VRAM (with LoRA, r=16)** | ~10-11 GB |
| **Context fill** (262k tokens) | ~30-40 sec (single GPU) |

---

## Key Papers & References

- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **Proportional RoPE**: (Gemma proprietary, 2025+)
- **Sliding Window Attention**: Child et al., "Efficient Transformers" (2019)
- **Group Query Attention**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- **Vision Encoding**: Gemma4 proprietary encoder-free design (2026)

---

## Links & Resources

- **HuggingFace Model Hub**: `google/gemma-4-12B-it`
- **Transformers Docs**: https://huggingface.co/docs/transformers
- **GitHub Code**: `transformers/src/transformers/models/gemma4_unified/`
- **License**: Apache 2.0 (Gemma weights are Google proprietary)

---

## Version Info

- **Gemma4 Unified Release**: 2026
- **Transformers Version**: 4.40.0+
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Documentation Date**: June 5, 2026

---

**Quick tip**: Save this card as a bookmarks file or print it out for reference during implementation!
