# Data Labels Loading & Storage Pipeline

## Data Flow: Masks → Labels → Model

### 1. **Initial Load** (`src/finetune_dataset.py`)

**Location**: `BCSSSegmentLoader.load(frame_id)`

```
data/bcss/masks/*.png
       ↓
   Image.open() → numpy array
       ↓
   np.unique() → extract object IDs (class labels)
       ↓
   for each object_id:
       - Binary segment created (uint8)
       - Prompts generated via get_prompts_from_mask()
```

**Key**: Mask files are palettized images where pixel values = object IDs/class labels

**Storage**: `self._last_prompts` dict (in memory)
```python
prompts_per_obj[obj_id] = {
    'point_coords': torch.float32,  # shape (N, 2)
    'point_labels': torch.int32,    # shape (N,)
}
```

---

### 2. **Segment Loading** (`src/finetune_dataset.py`)

**Location**: `BCSSSegmentLoader.get_prompts_for_last_load()`

Returns mapping of object IDs to their prompts:
```python
{
    1: {'point_coords': tensor(...), 'point_labels': tensor(...)},
    2: {'point_coords': tensor(...), 'point_labels': tensor(...)},
    ...
}
```

---

### 3. **VOSDataset Integration** (`sam2/training/dataset/vos_dataset.py`)

**Location**: `VOSDataset.construct()` lines 79-139

```
segment_loader.load(frame_idx)
       ↓
prompts_map = segment_loader.get_prompts_for_last_load()
       ↓
for each object_id in sampled_object_ids:
    segments[obj_id] → binary mask (torch.uint8)
    prompts_map[obj_id] → explicit prompts (point_coords, point_labels)
       ↓
    Object(
        object_id=obj_id,
        segment=segment,
        point_coords=point_coords,
        point_labels=point_labels,
    )
```

---

### 4. **Transforms** (`sam2/training/dataset/transforms.py`)

**Transforms adjust coordinates to match resized images**:
- `RandomHorizontalFlip`: Flip coordinates (if x in [0, w], new x = w - x)
- `RandomResizeAPI`: Scale coordinates by resize factor
- `ToTensorAPI`: Convert to tensors

**Prompt coordinates are preserved** (not lost during augmentation)

---

### 5. **Batching** (`sam2/training/utils/data_utils.py`)

**Location**: `collate_fn()` + `BatchedVideoDatapoint`

```
Object (single) → BatchedVideoDatapoint (batched)

Prompts per frame padded to shape:
  [T, O, K, 2]  where T=frames, O=objects, K=points
  
Labels padded to shape:
  [T, O, K]
```

**Dataclass fields**:
```python
class BatchedVideoDatapoint:
    point_coords: torch.Tensor  # [T, O, K, 2]
    point_labels: torch.Tensor  # [T, O, K]
    masks: torch.Tensor         # [T, O, H, W] (binary masks)
```

---

### 6. **Training Model** (`sam2/training/model/sam2.py`)

**Location**: `SAM2Train.forward()`

When `explicit_prompt_type` is set:
- Uses `batch.point_coords` and `batch.point_labels` directly
- Bypasses random prompt sampling
- Passes to SAM2 decoder during forward pass

```python
if self.explicit_prompt_type is not None:
    # Use dataset-provided prompts
    prompts = {
        'point_coords': batch.point_coords,
        'point_labels': batch.point_labels,
    }
else:
    # Generate random prompts (default)
    prompts = sample_random_prompts(...)
```

---

### 7. **Loss Computation**

**Location**: `trainer.py` → `_step()` → Loss function

```
Model output (logits, IoU predictions)
       ↓
Loss function (MultiStepMultiMasksAndIous)
       ↓
Compares predictions vs batch.masks (ground-truth binary masks)
```

**Masks are never saved externally** during training. They exist only in:
- Memory (as tensors in batch)
- Checkpoints (encoder/decoder weights, not data)

---

### 8. **Checkpoint Saving** (`sam2/training/trainer.py`)

**Location**: `save_checkpoint()` lines 340-380

Saved per epoch to `finetune_logs/checkpoints/checkpoint.pt`:
```python
checkpoint = {
    'model': state_dict,           # ← Model weights only
    'optimizer': optimizer_state,
    'epoch': epoch,
    'loss': loss_state,
    'steps': steps,
    'time_elapsed': elapsed,
    'best_meter_values': meters,
}
```

**Labels/masks NOT saved in checkpoint** (only model parameters saved)

---

## Summary: Where Data Flows

| Stage | Input | Output | Storage |
|-------|-------|--------|---------|
| **Load** | `data/bcss/masks/*.png` | Object IDs, binary segments | Memory |
| **Prompts** | Binary segments | point_coords, point_labels | Memory |
| **VOSDataset** | Prompts + segments | Object instances | Memory |
| **Transforms** | Coordinates | Adjusted coordinates | Memory |
| **Collate** | Objects → Batch | Padded tensors [T,O,K,2] | Memory |
| **Model** | Batch prompts | Loss gradients | GPU memory |
| **Save** | Model weights | Checkpoint .pt file | `finetune_logs/checkpoints/` |

---

## Key File Locations

| File | Purpose | Line |
|------|---------|------|
| `src/finetune_dataset.py` | Load masks, generate prompts | 26-75 |
| `sam2/training/dataset/vos_dataset.py` | Attach prompts to objects | 79-139 |
| `sam2/training/utils/data_utils.py` | Batch prompts with padding | collate_fn |
| `sam2/training/dataset/transforms.py` | Transform prompt coordinates | hflip/resize/pad |
| `sam2/training/model/sam2.py` | Use explicit prompts in forward | forward() |
| `sam2/training/trainer.py` | Save model checkpoint | 340-380 |

