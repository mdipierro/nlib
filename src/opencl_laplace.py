"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
from random import choice, randint

import numpy
from canvas import Canvas
from device import Device

n = 300
q = numpy.zeros((n, n), dtype=numpy.float32)
u = numpy.zeros((n, n), dtype=numpy.float32)
w = numpy.zeros((n, n), dtype=numpy.float32)

for k in range(n):
    q[randint(1, n - 1), randint(1, n - 1)] = choice((-1, +1))

device = Device()
q_buffer = device.buffer(source=q, mode=device.flags.READ_ONLY)
u_buffer = device.buffer(source=u)
w_buffer = device.buffer(source=w)


kernels = device.compile(
    """
       __kernel void solve(__global float * w,
                           __global const float * u,
                           __global const float * q) {
          int x = get_global_id(0);
          int y = get_global_id(1);
          int xy = y * WIDTH + x, up, down, left, right;
          if(y!=0 && y!=WIDTH-1 && x!=0 && x!=WIDTH-1) {
             up=xy+WIDTH; down=xy-WIDTH; left=xy-1; right=xy+1;
             w[xy] = 1.0/4.0 * (u[up]+u[down]+u[left]+u[right] - q[xy]);
          }
       }
    """.replace(
        "WIDTH", str(n)
    )
)

for k in range(1000):
    kernels.solve(device.queue, [n, n], None, w_buffer, u_buffer, q_buffer)
    (u_buffer, w_buffer) = (w_buffer, u_buffer)

u = device.retrieve(u_buffer, shape=(n, n))

Canvas().imshow(u).save(filename="plot.png")
