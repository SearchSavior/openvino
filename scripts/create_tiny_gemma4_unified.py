#!/usr/bin/env python3
"""
Create a tiny Gemma4 Unified model with random weights in bfloat16 precision.

This script creates a minimal version of the model suitable for:
- Testing and debugging
- Quick turnaround development
- VRAM-limited environments
- Integration testing

Model specs:
- Text: 2 layers, 128 hidden, 4 heads (2 KV), 512 vocab
- Vision: 16px patch size, 3x3 pooling, 128 embed dim
- Audio: 640-sample frames, 128 embed dim
- Precision: bfloat16
- Estimated VRAM: ~256 MB
"""

import os
import torch
from pathlib import Path

# Ensure transformers is available
try:
    from transformers import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedVisionConfig,
        Gemma4UnifiedAudioConfig,
        Gemma4UnifiedForConditionalGeneration,
        GemmaTokenizer,
    )
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure transformers is installed: pip install transformers")
    exit(1)


def create_tiny_gemma4_unified(
    output_dir: str = "./tiny-gemma4-unified",
    vocab_size: int = 512,
    hidden_size: int = 128,
    num_layers: int = 12,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Create a tiny Gemma4 Unified model with random weights.

    Args:
        output_dir: Directory to save the model
        vocab_size: Vocabulary size (small for tiny model)
        hidden_size: Hidden dimension (small for tiny model)
        num_layers: Number of decoder layers (12 for tiny; exercises 5:1 sliding/full pattern)
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (group query attention)
        dtype: Torch dtype (bfloat16 recommended)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating tiny Gemma4 Unified model...")
    print(f"  Output directory: {output_path.absolute()}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Precision: {dtype}")

    # === Create Text Config ===
    # Generate 5:1 sliding/full pattern for num_layers
    # Pattern: 5 sliding + 1 full = 6-layer repeat
    sliding_window_pattern = 6
    layer_types = [
        "sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"
        for i in range(num_layers)
    ]
    # Force last layer to full_attention
    layer_types[-1] = "full_attention"

    text_config = Gemma4UnifiedTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,  # Standard: 4x hidden
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        max_position_embeddings=2048,  # Small context for tiny model
        sliding_window=256,  # Small window
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        use_bidirectional_attention="vision",
        layer_types=layer_types,  # Properly distributed 5:1 pattern
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
    )

    # === Create Vision Config ===
    vision_config = Gemma4UnifiedVisionConfig(
        patch_size=16,
        pooling_kernel_size=3,
        mm_embed_dim=hidden_size,
        mm_posemb_size=128,  # Sized for ~64-100 patches
        output_proj_dims=hidden_size,
        rms_norm_eps=1e-6,
    )

    # === Create Audio Config ===
    audio_config = Gemma4UnifiedAudioConfig(
        audio_embed_dim=640,  # Standard: raw audio frame size
        rms_norm_eps=1e-6,
    )

    # === Create Top-Level Config ===
    config = Gemma4UnifiedConfig(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        boi_token_id=255_999,
        eoi_token_id=258_882,
        image_token_id=258_880,
        boa_token_id=256_000,
        eoa_token_index=258_883,
        audio_token_id=258_881,
        video_token_id=258_884,
    )

    print(f"\nConfig created:")
    print(f"  Text: {num_layers} layers, {hidden_size} hidden, {num_heads} heads")
    print(f"  Vision: patch={vision_config.patch_size}, pool={vision_config.pooling_kernel_size}")
    print(f"  Audio: {audio_config.audio_embed_dim} samples/token @ 16kHz")

    # === Create Model ===
    print(f"\nInitializing model with random weights...")
    model = Gemma4UnifiedForConditionalGeneration(config)

    # Convert to dtype
    model = model.to(dtype=dtype)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Estimate memory usage (bytes)
    if dtype == torch.bfloat16:
        bytes_per_param = 2  # 16-bit
    elif dtype == torch.float16:
        bytes_per_param = 2
    else:  # float32
        bytes_per_param = 4

    estimated_mb = (total_params * bytes_per_param) / (1024 ** 2)
    print(f"  Estimated VRAM: {estimated_mb:.1f} MB")

    # === Save Model ===
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    print(f"✅ Model saved successfully!")

    # === Create Minimal Tokenizer ===
    # We'll create a simple tokenizer file (dummy, just for compatibility)
    print(f"\nNote: A real tokenizer (SentencePiece .model file) is required for actual use.")
    print(f"      Use: transformers-cli download google/gemma-4-12B-it --token <your-token>")

    # === Create Config.json Details ===
    print(f"\n📝 Saved files:")
    print(f"  - config.json")
    print(f"  - model.safetensors")
    print(f"  - generation_config.json")

    # === Quick Test ===
    print(f"\n🧪 Quick validation...")
    try:
        # Test that we can load the model
        from transformers import Gemma4UnifiedForConditionalGeneration as G4U
        test_model = G4U.from_pretrained(str(output_path))
        test_model = test_model.to(dtype=dtype)

        # Test forward pass with dummy input
        with torch.no_grad():
            dummy_input_ids = torch.randint(0, vocab_size, (1, 16))
            outputs = test_model(input_ids=dummy_input_ids)

        print(f"  ✓ Model loads successfully")
        print(f"  ✓ Forward pass works")
        print(f"  ✓ Output logits shape: {outputs.logits.shape}")
        print(f"  ✓ Output dtype: {outputs.logits.dtype}")

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")

    print(f"\n{'='*60}")
    print(f"Model ready for use!")
    print(f"{'='*60}")

    return output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a tiny Gemma4 Unified model with random weights"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tiny-gemma4-unified",
        help="Output directory for the model (default: ./tiny-gemma4-unified)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Vocabulary size (default: 512)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of decoder layers (default: 12, exercises 5:1 sliding/full pattern)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Create model
    create_tiny_gemma4_unified(
        output_dir=args.output,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
