"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import numpy as npy
from device import Device

n = 100000
u = npy.random.rand(n).astype(npy.float32)
v = npy.random.rand(n).astype(npy.float32)

device = Device()
u_buffer = device.buffer(source=u)
v_buffer = device.buffer(source=v)
w_buffer = device.buffer(size=v.nbytes)

kernels = device.compile(
    """
    __kernel void sum(__global const float * u,
                      __global const float * v,
                      __global float * w) {
      int i = get_global_id(0);
      w[i] = u[i] + v[i];
    }
    """
)

kernels.sum(device.queue, [n], None, u_buffer, v_buffer, w_buffer)
w = device.retrieve(w_buffer)

assert npy.linalg.norm(w - (u + v)) == 0
