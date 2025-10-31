# FakeInferenceEngine Offloading Support

## Summary

Updated `FakeInferenceEngine` to support offloading configuration and generate realistic fake offloading metrics matching the real `InferenceEngine` format.

## Changes Made

### 1. Updated `InferenceBase.create()` (Lines 94-125)
- Moved offload config parsing before engine instantiation
- Now passes `offload_config_obj` to both `FakeInferenceEngine` and `InferenceEngine`
- Supports both preset mode and manual mode

### 2. Updated `FakeInferenceEngine.__init__()` (Lines 330-345)
- Added `offload_config` parameter (optional)
- Stores config for use in metric generation
- Mirrors `InferenceEngine` constructor signature

### 3. Added `_generate_fake_offloading_metrics()` (Lines 355-421)
- Generates realistic fake metrics based on CLAUDE.md Phase 2 measurements
- Uses target overhead percentages:
  - **Balanced** (12+ GPU layers): 75% overhead
  - **Aggressive** (8-11 GPU layers): 82% overhead
  - **Extreme** (4-7 GPU layers): 87% overhead
- Distributes time across CPU→GPU transfer, GPU→CPU release, and computation
- Calculates theoretical async savings (~90% of GPU→CPU time)
- Estimates VRAM savings (0.31 GB per CPU layer for Float8)

### 4. Updated `_save_audio()` (Lines 423-455)
- Calls `_generate_fake_offloading_metrics()` when offloading enabled
- Stores metrics in `generation.details['offloading_metrics']`
- Logs metrics with [FAKE] prefix for clarity
- Maintains same structure as real engine

## Fake Metrics Structure

```python
{
    "enabled": True,
    "gpu_layers": 12,           # Based on config
    "cpu_layers": 16,            # 28 - gpu_layers
    "transfer_overhead_ms": 37500.0,
    "avg_layer_transfer_ms": 36.8,
    "overhead_percentage": 75.0,
    "time_breakdown": {
        "pure_computation_ms": 12500.0,
        "cpu_to_gpu_transfer_ms": 19259.51,
        "gpu_to_cpu_release_ms": 18240.49,
    },
    "theoretical_async_savings_ms": 16416.44,
    "vram_saved_gb": 4.96,
}
```

## Test Results

```
Balanced Preset (12 GPU layers):
  - VRAM Saved: 4.96 GB
  - Overhead: 75.00%
  - Transfer Time: 37.5s (for 50s generation)

Aggressive Preset (8 GPU layers):
  - VRAM Saved: 6.2 GB
  - Overhead: 82.00%
  - Transfer Time: 41.0s (for 50s generation)

Extreme Preset (4 GPU layers):
  - VRAM Saved: 7.44 GB
  - Overhead: 87.00%
  - Transfer Time: 43.5s (for 50s generation)
```

## Usage Example

```python
# In API request
offload_config = {
    "enabled": True,
    "mode": "preset",
    "preset": "balanced"
}

# Engine creation (automatically handles fake vs real)
engine = InferenceBase.create(
    generation=generation,
    speaker_service=speaker_service,
    dialog_service=dialog_service,
    meta_file_path=meta_file_path,
    fake=True,  # Use FakeInferenceEngine
    offload_config=offload_config
)

# After generation completes
metrics = generation.details.get('offloading_metrics')
# metrics will contain realistic fake data matching real engine format
```

## Frontend Compatibility

The fake metrics are **100% compatible** with the frontend components:
- `CurrentGeneration.tsx` displays offloading status
- `GenerationHistory.tsx` shows full metrics breakdown
- Same field names and structure as real engine
- All type definitions (`OffloadingMetrics`) match

## Benefits

1. ✅ **Testing without GPU**: Can test offloading feature end-to-end without real hardware
2. ✅ **UI Development**: Frontend developers can see realistic metrics during development
3. ✅ **Consistent API**: Same interface for both fake and real engines
4. ✅ **Realistic Values**: Metrics based on real measurements from CLAUDE.md
5. ✅ **Type Safety**: Full TypeScript compatibility

## Implementation Notes

- Overhead percentages calibrated from CLAUDE.md Phase 2 metrics
- Transfer time ratio (CPU→GPU : GPU→CPU) ≈ 51.4% : 48.6% (based on 37.8ms : 35.8ms)
- VRAM savings estimated at 310 MB per layer (Float8 weights)
- Theoretical async savings assumes 90% of GPU→CPU transfers can be hidden
- All metrics rounded to 2 decimal places for consistency

## Files Modified

- `backend/inference/inference.py`:
  - Lines 94-125: Updated `InferenceBase.create()`
  - Lines 330-345: Updated `FakeInferenceEngine.__init__()`
  - Lines 355-421: Added `_generate_fake_offloading_metrics()`
  - Lines 423-455: Updated `_save_audio()`

## Testing

Verified with:
```bash
python -m py_compile backend/inference/inference.py  # ✓ No syntax errors
python test_fake_offloading.py                       # ✓ Realistic metrics generated
```

## Date

2025-10-31
