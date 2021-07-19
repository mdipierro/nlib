"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import numpy
import pyopencl as pcl


class Device(object):
    flags = pcl.mem_flags

    def __init__(self):
        self.ctx = pcl.create_some_context()
        self.queue = pcl.CommandQueue(self.ctx)

    def buffer(self, source=None, size=0, mode=pcl.mem_flags.READ_WRITE):
        if source is not None:
            mode = mode | pcl.mem_flags.COPY_HOST_PTR
        buffer = pcl.Buffer(self.ctx, mode, size=size, hostbuf=source)
        return buffer

    def retrieve(self, buffer, shape=None, dtype=numpy.float32):
        output = numpy.zeros(shape or buffer.size // 4, dtype=dtype)
        pcl.enqueue_copy(self.queue, output, buffer)
        return output

    def compile(self, kernel):
        return pcl.Program(self.ctx, kernel).build()
