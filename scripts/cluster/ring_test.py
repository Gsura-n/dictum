"""Sanity check for the cluster: every rank all-sums a ones vector and reports
throughput for a LoRA-sized gradient (about 28 MB) over the ring."""
import socket
import time

import mlx.core as mx

world = mx.distributed.init(backend="ring")
x = mx.distributed.all_sum(mx.ones(4))
mx.eval(x)
payload = mx.ones((7_000_000,), dtype=mx.float32)   # ~28 MB, same as one LoRA gradient step
mx.eval(payload)
t0 = time.perf_counter()
for _ in range(10):
    y = mx.distributed.all_sum(payload)
    mx.eval(y)
dt = (time.perf_counter() - t0) / 10
print(f"rank {world.rank()}/{world.size()} on {socket.gethostname()}: all_sum ok ({x.tolist()}), "
      f"28 MB gradient sync {dt * 1000:.0f} ms", flush=True)
