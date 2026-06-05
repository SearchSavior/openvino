# Gemma4 Unified: Practical Code Examples

This document provides concrete code examples for loading, using, and implementing the Gemma4 Unified model.

---

## Table of Contents

1. [Loading and Basic Usage](#loading-and-basic-usage)
2. [Processing Multimodal Inputs](#processing-multimodal-inputs)
3. [Generation & Inference](#generation--inference)
4. [Training & Fine-Tuning](#training--fine-tuning)
5. [Key Component Implementation](#key-component-implementation)

---

## Loading and Basic Usage

### Python with HuggingFace Transformers

```python
import torch
from transformers import (
    Gemma4UnifiedProcessor,
    Gemma4UnifiedForConditionalGeneration,
)

# Load model and processor
model_id = "google/gemma-4-12B-it"
processor = Gemma4UnifiedProcessor.from_pretrained(model_id)
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # or float16
    device_map="auto",
)

print(f"Model loaded. Config:")
print(f"  Text hidden size: {model.config.text_config.hidden_size}")
print(f"  Num layers: {model.config.text_config.num_hidden_layers}")
print(f"  Context length: {model.config.text_config.max_position_embeddings}")
```

### Configuration Inspection

```python
from transformers import Gemma4UnifiedConfig

config = Gemma4UnifiedConfig.from_pretrained("google/gemma-4-12B-it")

# Text config
text_cfg = config.text_config
print(f"Layer types: {text_cfg.layer_types[:6]}...")  # [sliding, sliding, ..., full]
print(f"Sliding window: {text_cfg.sliding_window}")
print(f"Bidirectional attention: {text_cfg.use_bidirectional_attention}")

# Vision config
vision_cfg = config.vision_config
print(f"Patch size: {vision_cfg.patch_size}")
print(f"Model patch size: {vision_cfg.model_patch_size}")
print(f"Positional embedding size: {vision_cfg.mm_posemb_size}")

# Audio config
audio_cfg = config.audio_config
print(f"Audio samples per token: {audio_cfg.audio_samples_per_token}")

# Special tokens
print(f"Image token: {config.image_token_id}")
print(f"Audio token: {config.audio_token_id}")
print(f"BOI: {config.boi_token_id}, EOI: {config.eoi_token_id}")
```

---

## Processing Multimodal Inputs

### 1. Image Processing

```python
from PIL import Image
import numpy as np

# Load image
image = Image.open("path/to/image.jpg")

# Process single image
outputs = processor.image_processor(
    images=[image],
    max_soft_tokens=280,  # 70, 140, 280, 560, 1120
    return_tensors="pt",
)

print(f"Pixel values shape: {outputs['pixel_values'].shape}")
# → (1, 280, 6912)

print(f"Image position IDs shape: {outputs['image_position_ids'].shape}")
# → (1, 280, 2)

print(f"Num soft tokens per image: {outputs['num_soft_tokens_per_image']}")
# → [actual_count]
```

### 2. Video Processing

```python
import cv2
import numpy as np

# Load video frames
cap = cv2.VideoCapture("path/to/video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()

# Convert to tensor (num_frames, H, W, 3)
video_tensor = torch.tensor(np.stack(frames), dtype=torch.uint8)
# or np.ndarray: (num_frames, H, W, 3)

# Process video
outputs = processor.video_processor(
    videos=[video_tensor],
    num_frames=32,  # sample this many frames
    max_soft_tokens=70,  # lower for videos due to temporal cost
    return_tensors="pt",
)

print(f"Video pixel values shape: {outputs['pixel_values_videos'].shape}")
# → (batch, num_sampled_frames, 70, 6912)
```

### 3. Audio Processing

```python
import librosa
import numpy as np

# Load audio (16 kHz)
audio_path = "path/to/audio.wav"
waveform, sr = librosa.load(audio_path, sr=16_000)

# Or use soundfile
import soundfile as sf
waveform, sr = sf.read(audio_path)
# Resample if needed
if sr != 16_000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16_000)

# Process audio
outputs = processor.feature_extractor(
    raw_speech=[waveform],
    sampling_rate=16_000,
    padding="longest",
    truncation=True,
    max_length=None,  # or set max frames
    return_tensors="pt",
)

print(f"Input features shape: {outputs['input_features'].shape}")
# → (batch, max_frames, 640)

print(f"Input features mask shape: {outputs['input_features_mask'].shape}")
# → (batch, max_frames)

print(f"Num frames: {outputs['input_features'].shape[1]}")
```

### 4. Unified Processing: All Modalities

```python
from PIL import Image
import torch

# Prepare inputs
image = Image.open("image.jpg")
text_prompt = "Describe this image."

# Single call processes everything
inputs = processor(
    images=[image],
    text=text_prompt,
    padding=True,
    return_tensors="pt",
)

print("Input keys:", inputs.keys())
# → ['input_ids', 'attention_mask', 'pixel_values', 'image_position_ids',
#    'num_soft_tokens_per_image', 'mm_token_type_ids']

# Multimodal: image + text
inputs = processor(
    images=[[image_1, image_2]],  # nested list (batch of image lists)
    text=["Describe these.", "Compare them."],
    padding=True,
    return_tensors="pt",
)

# Audio + text
import librosa
audio, sr = librosa.load("audio.wav", sr=16_000)

inputs = processor(
    audio=[audio],
    text=["Transcribe this."],
    padding=True,
    return_tensors="pt",
)

# Video + text
video_frames = np.random.randint(0, 256, (32, 480, 640, 3), dtype=np.uint8)
inputs = processor(
    videos=[video_frames],
    text=["Describe the video."],
    padding=True,
    return_tensors="pt",
)

# Move to device
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
          for k, v in inputs.items()}
```

---

## Generation & Inference

### 1. Basic Generation

```python
import torch
from transformers import Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedProcessor

model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "google/gemma-4-12B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = Gemma4UnifiedProcessor.from_pretrained("google/gemma-4-12B-it")

# Prepare input
from PIL import Image
image = Image.open("image.jpg")

inputs = processor(
    images=[image],
    text="Describe this image in detail:",
    return_tensors="pt",
)

# Move to device
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
          for k, v in inputs.items()}

# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# Decode
response = processor.tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
)
print(response)
```

### 2. Streaming Generation

```python
from transformers import TextIteratorStreamer, Gemma4UnifiedForConditionalGeneration
from threading import Thread

model = Gemma4UnifiedForConditionalGeneration.from_pretrained(...)
processor = Gemma4UnifiedProcessor.from_pretrained(...)

inputs = processor(
    images=[image],
    text="Describe this image:",
    return_tensors="pt",
).to(model.device)

# Setup streamer
streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_special_tokens=True,
)

# Generate in thread
generation_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_new_tokens=200,
    temperature=0.7,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream tokens
for text in streamer:
    print(text, end="", flush=True)

thread.join()
```

### 3. Batch Generation

```python
from PIL import Image
import torch

# Prepare batch
images = [
    Image.open("image1.jpg"),
    Image.open("image2.jpg"),
    Image.open("image3.jpg"),
]

texts = [
    "Describe this image:",
    "What's in this image?",
    "Summarize this image:",
]

# Process batch
inputs = processor(
    images=[images],  # nested list for batch
    text=texts,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Generate
output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
)

# Decode all
responses = processor.tokenizer.batch_decode(
    output_ids,
    skip_special_tokens=True,
)

for i, response in enumerate(responses):
    print(f"Image {i}: {response}")
```

### 4. Custom Generation Parameters

```python
# Nucleus sampling (top-p)
output_ids = model.generate(
    **inputs,
    do_sample=True,
    top_p=0.95,
    temperature=0.8,
    max_new_tokens=100,
)

# Top-K sampling
output_ids = model.generate(
    **inputs,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    max_new_tokens=100,
)

# Greedy (deterministic)
output_ids = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=100,
)

# Beam search
output_ids = model.generate(
    **inputs,
    num_beams=4,
    num_beam_groups=1,
    max_new_tokens=100,
)

# Diverse beam search
output_ids = model.generate(
    **inputs,
    num_beams=4,
    num_beam_groups=2,
    diversity_penalty=0.5,
    max_new_tokens=100,
)
```

---

## Training & Fine-Tuning

### 1. Loading for Training

```python
import torch
from transformers import (
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedProcessor,
)
from peft import get_peft_model, LoraConfig, TaskType

# Load base model
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "google/gemma-4-12B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Option 1: LoRA (parameter-efficient fine-tuning)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj",  # attention
        "gate_proj", "up_proj", "down_proj",  # FFN
    ],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Option 2: Full fine-tuning (requires more VRAM)
for param in model.parameters():
    param.requires_grad = True
```

### 2. Data Preparation

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

class Gemma4UnifiedDataset(Dataset):
    def __init__(self, data_file, processor, max_tokens_per_image=280):
        """
        data_file: JSON with format:
        [
            {
                "image": "path/to/image.jpg",
                "text": "Question?",
                "response": "Answer."
            },
            ...
        ]
        """
        with open(data_file) as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.max_tokens_per_image = max_tokens_per_image
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item["image"]).convert("RGB")
        
        # Construct prompt
        text = f"{item['text']}\n{item['response']}"
        
        # Process
        inputs = self.processor(
            images=[image],
            text=text,
            max_soft_tokens=self.max_tokens_per_image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Flatten batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        
        # Labels (for loss computation) = input_ids shifted right
        inputs["labels"] = inputs["input_ids"].clone()
        # Mask out image token positions
        image_mask = inputs.get("pixel_values") is not None
        if image_mask:
            # Don't compute loss on image embeddings
            pass
        
        return inputs

# Create dataset and loader
dataset = Gemma4UnifiedDataset("data.json", processor)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=default_data_collator,  # from transformers
)

for batch in loader:
    print(batch.keys())
    # → ['input_ids', 'attention_mask', 'labels', 'pixel_values', ...]
    break
```

### 3. Training Loop

```python
import torch
from torch.optim import AdamW
from transformers import get_scheduler

# Model and optimizer
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "google/gemma-4-12B-it",
    torch_dtype=torch.bfloat16,
)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
num_epochs = 3
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)

# Training
device = "cuda:0"
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"  Validation Loss = {avg_val_loss:.4f}")
```

### 4. Distributed Training (Multi-GPU)

```python
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# Initialize distributed training
init_process_group(backend="nccl")

# Create model (on correct GPU)
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "google/gemma-4-12B-it",
    torch_dtype=torch.bfloat16,
)
model = DDP(model, device_ids=[rank], output_device=rank)

# Distributed sampler
train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=train_sampler,
)

# Training (same as above)
# optimizer updates on model.module, not model

# Cleanup
destroy_process_group()
```

---

## Key Component Implementation

### 1. Patches Merge (Python)

```python
import torch
import math

def patches_merge(
    patches: torch.Tensor,
    positions_xy: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge k×k groups of teacher patches into model patches.
    
    Args:
        patches: (*, L, D) where D = patch_size² × 3
        positions_xy: (*, L, 2) integer XY positions
        target_length: output number of patches
    
    Returns:
        merged_patches: (*, target_length, k²×D)
        merged_positions: (*, target_length, 2)
    """
    patch_size = math.isqrt(patches.shape[-1] // 3)
    k = math.isqrt(patches.shape[-2] // target_length)
    
    if k * k * target_length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {target_length}")
    
    # Compute spatial grouping
    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = (
        k * k * kernel_idxs[..., 0] +
        k * max_x * kernel_idxs[..., 1]
    )
    
    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[..., 0] +
        position_within_kernel[..., 1] * k
    )
    target_ordering = (
        num_patches_from_top_left_of_kernel +
        num_patches_from_top_left
    )
    
    # Permute patches into kernel-grouped order
    perm = target_ordering.long().argsort(dim=-1)
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)
    
    batch_shape = patches.shape[:-2]
    
    # Reshape into k×k blocks
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, target_length, k * k, patch_size, patch_size, 3
    )
    
    # Rearrange spatial dimensions
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, target_length, k, k, patch_size, patch_size, 3
    )
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )
    merged_patches = kernel_ordered_patches.reshape(
        *batch_shape, target_length, k * patch_size * k * patch_size * 3
    )
    
    # Update positions
    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())
    
    # Handle padding
    padding = (positions_xy == -1).all(dim=-1, keepdim=True)
    kernel_ordered_positions = (
        kernel_ordered_positions * (~padding).float() +
        positions_xy.float() * padding.float()
    )
    
    # Reshape and take min position
    kernel_ordered_positions = kernel_ordered_positions.reshape(
        *batch_shape, target_length, k * k, 2
    )
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)
    
    return merged_patches, new_positions

# Example usage
teacher_patches = torch.randn(280, 768)  # 280 teacher patches
positions = torch.arange(280).unsqueeze(-1).repeat(1, 2) % 20
# Create valid (x, y) positions

model_patches, model_positions = patches_merge(
    teacher_patches.unsqueeze(0),
    positions.unsqueeze(0),
    target_length=280 // 9,  # 3×3 merge
)

print(model_patches.shape)  # (1, 31, 6912)
print(model_positions.shape)  # (1, 31, 2)
```

### 2. Per-Layer Type RoPE (PyTorch)

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class Gemma4UnifiedRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns = {}
        self.rope_type = {}
        
        # Initialize RoPE for each layer type
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            rope_theta = rope_params["rope_theta"]
            rope_type = rope_params["rope_type"]
            
            # Get head dimension
            if layer_type == "full_attention" and rope_type == "proportional":
                head_dim = config.global_head_dim  # 512
            else:
                head_dim = config.head_dim  # 256
            
            # Compute inverse frequencies
            if rope_type == "default":
                # Standard RoPE: θ^(2i/d)
                inv_freq = 1.0 / (
                    rope_theta ** (
                        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) /
                        head_dim
                    )
                )
            elif rope_type == "proportional":
                # Proportional RoPE: uses partial_rotary_factor
                partial_factor = rope_params.get("partial_rotary_factor", 0.25)
                rotary_dim = int(head_dim * partial_factor)
                inv_freq = 1.0 / (
                    rope_theta ** (
                        torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) /
                        rotary_dim
                    )
                )
            else:
                raise ValueError(f"Unknown rope_type: {rope_type}")
            
            # Register buffers
            self.register_buffer(
                f"{layer_type}_inv_freq",
                inv_freq,
                persistent=False,
            )
            self.rope_type[layer_type] = rope_type
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, heads, head_dim)
        position_ids: torch.Tensor,  # (batch, seq_len)
        layer_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE for a specific layer type.
        
        Returns:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        
        # Expand for broadcasting
        inv_freq_expanded = inv_freq[None, :, None].float()  # (1, rotary_dim//2, 1)
        inv_freq_expanded = inv_freq_expanded.to(x.device)
        
        position_ids_expanded = position_ids[:, None, :].float()  # (batch, 1, seq_len)
        
        # Compute angles: θ * m (m = position)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # → (batch, seq_len, rotary_dim//2)
        
        # Concatenate to full dimension
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq_len, rotary_dim)
        
        # Compute cos and sin
        cos = emb.cos()  # (batch, seq_len, rotary_dim)
        sin = emb.sin()
        
        return cos, sin

# Example usage
config = type('obj', (object,), {
    'head_dim': 256,
    'global_head_dim': 512,
    'layer_types': ['sliding_attention', 'full_attention'],
    'rope_parameters': {
        'sliding_attention': {
            'rope_theta': 10_000.0,
            'rope_type': 'default',
        },
        'full_attention': {
            'rope_theta': 1_000_000.0,
            'rope_type': 'proportional',
            'partial_rotary_factor': 0.25,
        },
    }
})()

rope_emb = Gemma4UnifiedRotaryEmbedding(config)

# Forward pass
x = torch.randn(2, 512, 8, 256)  # (batch, seq_len, heads, head_dim)
position_ids = torch.arange(512).unsqueeze(0).expand(2, -1)

cos_full, sin_full = rope_emb(x, position_ids, "full_attention")
cos_sliding, sin_sliding = rope_emb(x, position_ids, "sliding_attention")

print(cos_full.shape)  # (2, 512, 512) or (2, 512, rotary_dim)
```

### 3. Attention Mask with Bidirectional Vision (Python)

```python
import torch

def create_bidirectional_vision_mask(
    seq_len: int,
    block_sequence_ids: torch.Tensor,  # (batch, seq_len) or None
    sliding_window: int = 1024,
    layer_type: str = "full_attention",
) -> torch.Tensor:
    """
    Create attention mask for bidirectional vision tokens.
    
    Args:
        seq_len: sequence length
        block_sequence_ids: -1=text, ≥0=vision/block id
        sliding_window: window size for sliding attention
        layer_type: "sliding_attention" or "full_attention"
    
    Returns:
        mask: (1, 1, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
              0 = attend, -inf = mask out
    """
    if block_sequence_ids is None:
        # Standard causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    batch_size = block_sequence_ids.shape[0]
    mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                block_i = block_sequence_ids[b, i].item()
                block_j = block_sequence_ids[b, j].item()
                
                # Same vision block: bidirectional
                if block_i >= 0 and block_i == block_j:
                    mask[b, 0, i, j] = 0.0
                # Different blocks or text: causal
                elif j > i:
                    mask[b, 0, i, j] = float('-inf')
                else:
                    mask[b, 0, i, j] = 0.0
                
                # Apply sliding window for sliding attention layers
                if layer_type == "sliding_attention" and j < i - sliding_window:
                    mask[b, 0, i, j] = float('-inf')
    
    return mask

# Example
batch_size, seq_len = 2, 512

# Tokens 0-279: vision (block_id=0)
# Tokens 280-511: text (block_id=-1)
block_sequence_ids = torch.full((batch_size, seq_len), -1)
block_sequence_ids[:, :280] = 0

mask = create_bidirectional_vision_mask(
    seq_len,
    block_sequence_ids,
    sliding_window=1024,
    layer_type="full_attention",
)

print(mask.shape)  # (2, 1, 512, 512)

# Vision tokens (0-279) can attend to all vision tokens and past text
# Text tokens (280+) can only attend to past tokens (causal)
print(mask[0, 0, 100, 200])  # vision→vision: 0.0 (attend)
print(mask[0, 0, 400, 100])  # text→vision: -inf (mask)
print(mask[0, 0, 400, 300])  # text→past_text: 0.0 (attend)
```

---

## Summary

These examples cover:
- **Loading & configuration**: How to load the model and inspect configs
- **Preprocessing**: Image, video, audio, and unified processing
- **Generation**: Basic, streaming, batched, and parameter customization
- **Training**: Dataset setup, training loops, distributed training
- **Implementation**: Key algorithms (patches_merge, RoPE, attention masking)

For more details, refer to the main architecture document: `GEMMA4_UNIFIED_ARCHITECTURE.md`

---

**Document Version**: 1.0 (2026)  
**Last Updated**: June 5, 2026
