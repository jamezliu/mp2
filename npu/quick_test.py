#!/usr/bin/env python3

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from conv_npu import conv2d
from conv_numpy import conv2d_cpu_torch
import torch

print("Quick test of modified conv2d with compiler allocation controls...")

# Test with MUCH smaller case for faster simulation
batch_size = 1
in_channels = 128
out_channels = 128
input_height = 2  # Much smaller
input_width = 2   # Much smaller
filter_height = 1 # Much smaller
filter_width = 1  # Much smaller

X = np.random.rand(batch_size, in_channels, input_height, input_width).astype(np.float32)
W = np.random.rand(out_channels, in_channels, filter_height, filter_width).astype(np.float32)
bias = np.random.rand(out_channels).astype(np.float32)

print(f"Input shape: {X.shape}")
print(f"Weight shape: {W.shape}")
print(f"Bias shape: {bias.shape}")

# Test simulation first
print("Testing simulation...")
try:
    out_sim = nki.simulate_kernel(conv2d, X, W, bias)
    print(f"Simulation output shape: {out_sim.shape}")
    print("✅ Simulation successful!")
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    exit(1)

# Test reference
print("Testing reference...")
try:
    out_ref = conv2d_cpu_torch(X, W, bias)
    print(f"Reference output shape: {out_ref.shape}")
    print("✅ Reference successful!")
except Exception as e:
    print(f"❌ Reference failed: {e}")
    exit(1)

# Compare results
print("Comparing results...")
if np.allclose(out_sim, out_ref, atol=1e-4):
    print("✅ Results match!")
else:
    print("❌ Results mismatch!")
    exit(1)

print("\n🎉 Quick test passed! Now testing hardware compilation...")

# Test hardware compilation (this is the key test)
print("Testing hardware compilation...")
try:
    # This will fail if TEN404 still occurs
    out_hw = conv2d(X, W, bias)
    print(f"Hardware output shape: {out_hw.shape}")
    print("🎉 HARDWARE COMPILATION SUCCESSFUL!")
    print("The compiler allocation controls worked!")
except Exception as e:
    print(f"❌ Hardware compilation still failed: {e}")
    if "TEN404" in str(e):
        print("TEN404 error still present - compiler allocation controls didn't work")
    else:
        print("Different error - might be progress!")

print("\nTest completed!")