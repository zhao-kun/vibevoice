# Diagnostic Profiling Guide

## The Problem

You're seeing **6-7 seconds per token on RTX 4090 with 20 GPU layers** - this is way too slow!

With only 8 layers offloaded on a 4090, you should be seeing **under 1 second per token**.

## Diagnose the Bottleneck

I've added detailed profiling to identify where the time is being spent. Run with `--profile` flag:

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_hero.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --num-gpu-layers 20 \
    --profile
```

## What the Profiling Shows

### Every 10 tokens (or every 5 seconds), you'll see:

```
================================================================================
PROFILING SUMMARY - Token 10
================================================================================
Async usage: 7/8 layers (87%)
  Layer  0: Pre= 45.23ms  Post=  0.12ms  [ASYNC]
  Layer  1: Pre= 42.18ms  Post=  0.10ms  [ASYNC]
  Layer  2: Pre= 43.55ms  Post=  0.11ms  [ASYNC]
  Layer  3: Pre=350.42ms  Post=  0.09ms  [SYNC]   <-- PROBLEM!
  ...
```

### Key Metrics to Look For:

1. **Async usage percentage**
   - **Good**: 100% (all layers use async)
   - **Bad**: <50% (mostly synchronous transfers)
   - **If low**: Async isn't working!

2. **Pre-transfer times**
   - **Good**: <100ms (waiting for prefetched layer)
   - **Bad**: >300ms (synchronous blocking transfer)
   - **If high**: Layer wasn't prefetched

3. **SYNC vs ASYNC status**
   - **Good**: All layers show [ASYNC]
   - **Bad**: Many layers show [SYNC]
   - **If SYNC**: Prefetching failed

4. **Post-transfer times**
   - **Good**: <1ms (async submit, no wait)
   - **Bad**: >200ms (synchronous blocking)
   - **If high**: Async CPU offload not working

## Likely Bottlenecks

### Bottleneck 1: Async Not Actually Running

**Symptoms**:
- All layers show `[SYNC]`
- Pre-transfer times >300ms
- Async usage: 0%

**Cause**: ThreadPoolExecutor might not be initialized

**Check in output**:
```
Async transfer: True  <-- Should see this
```

### Bottleneck 2: First Layer Always Synchronous

**Symptoms**:
- Layer 0 always shows `[SYNC]`
- Other layers show `[ASYNC]`
- First token slow, rest fast

**Cause**: First layer not prefetched (expected behavior)

**Solution**: This is normal, not a bottleneck

### Bottleneck 3: Model Compute Time

**Symptoms**:
- Async usage: 100%
- Pre-transfer times: <50ms
- But still 6-7s per token

**Cause**: **The model itself is slow**, not offloading!

**Check**: Run without offloading to get baseline:
```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_hero.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --no-offload \
    --profile
```

If this is also 6-7s/token, the problem is NOT offloading!

### Bottleneck 4: Diffusion Head

**Symptoms**:
- Transformer layers fast (<1s total)
- But overall token time still 6-7s

**Cause**: Diffusion head might be the bottleneck

**Check**: Look at total time breakdown in profiling

## Expected Timing Breakdown (RTX 4090, 20 GPU layers)

### Per Token (Target: <1s):
- Transformer layers (28 total):
  - 20 on GPU: 20 × 5ms compute = **100ms**
  - 8 offloaded:
    - Transfer wait: 8 × 30ms = **240ms** (async, prefetched)
    - Compute: 8 × 5ms = **40ms**
    - CPU offload: 8 × 0ms = **0ms** (async, background)
  - **Total transformer: ~400ms**

- Diffusion operations: **variable** (could be 1-5 seconds!)
- Other overhead: **~100ms**

**Expected total: 0.5-6 seconds** (depends on diffusion)

## Diagnostic Steps

### Step 1: Check Async is Working

Run with `--profile` and look for:
```
Async usage: X/8 layers (Y%)
```

- If Y < 50%: Async is broken
- If Y = 100%: Async is working

### Step 2: Check Transfer Times

Look at Pre-transfer times:
- <100ms = **GOOD** (async working)
- >300ms = **BAD** (sync blocking)

### Step 3: Isolate the Bottleneck

Run these tests:

```bash
# Test 1: With offloading + profiling
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --num-gpu-layers 20 \
    --profile

# Test 2: Without offloading (baseline)
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --no-offload

# Test 3: Fewer layers (verify scaling)
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --num-gpu-layers 24 \
    --profile
```

Compare times:
- If Test1 ≈ Test2: Offloading overhead is minimal (GOOD)
- If Test1 >> Test2: Offloading is broken
- If Test3 < Test1: Offloading works, just need more GPU layers

### Step 4: Check Diffusion Head

The diffusion head might be the actual bottleneck! With 20 layers on GPU, transformer should be <500ms, but diffusion could be 5+ seconds.

**To verify**: Look at the generation progress bar timing between "Step N" prints. If there's a big delay AFTER all transformer layers, it's the diffusion head.

## Report Back

Please run with `--profile` and share:

1. **Async usage percentage**
2. **A few lines of layer timing** (especially layers 0, 10, 19)
3. **Whether you see [SYNC] or [ASYNC]**
4. **Time per token with `--no-offload`** (baseline)

This will tell us exactly where the bottleneck is!

## Quick Test

To quickly identify the issue:

```bash
# Run just 10 tokens with profiling
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/zh-007_man.wav demo/voices/zh-007_woman.wav \
    --dtype bfloat16 \
    --num-gpu-layers 20 \
    --profile
```

**Kill after first profiling summary** (after 10 tokens) and share the output!
