import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1
    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0
    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Simplified tiled approach - 32-wide blocks with individual element operations
    C_OUT_B = 32  # Block size for output channels
    C_IN_B = 32   # Block size for input channels
    
    for b in nl.affine_range(batch_size):
        for out_h in nl.affine_range(out_height):
            for out_w in nl.affine_range(out_width):
                # Process each output channel individually (avoid partition issues)
                for c_out in nl.affine_range(out_channels):
                    # Initialize scalar accumulator
                    acc = nl.zeros((1, 1), dtype=X.dtype, buffer=nl.sbuf)
                    
                    # Process input channels in blocks of 32
                    for c_in_blk in nl.affine_range(0, in_channels, C_IN_B):
                        # Process filter positions
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                input_h = out_h + i
                                input_w = out_w + j
                                
                                # Process each input channel in the block
                                for c_in_idx in nl.affine_range(C_IN_B):
                                    actual_c_in = c_in_blk + c_in_idx
                                    
                                    # Load individual elements
                                    x_val = nl.load(X[b, actual_c_in, input_h, input_w])
                                    w_val = nl.load(W[c_out, actual_c_in, i, j])
                                    
                                    # Accumulate
                                    acc[0, 0] += x_val * w_val
                    
                    # Add bias
                    bias_val = nl.load(bias[c_out])
                    acc[0, 0] += bias_val
                    
                    # Store result
                    nl.store(X_out[b, c_out, out_h, out_w], acc[0, 0])

    return X_out

