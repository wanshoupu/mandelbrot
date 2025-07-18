import time
import numpy as np
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()  # call once, outside functions

def plain():
    start = time.time()
    a = np.random.rand(1_000_000)
    b = np.sin(a) * a + 2
    end = time.time()
    return end - start

def tensor():
    # Warmup (compile/tracing happens here)
    start = time.time()
    a = tnp.random.rand(1_000_000)
    _ = tnp.sin(a) * a + 2

    # Timed run
    b = tnp.sin(a) * a + 2
    end = time.time()
    return end - start

if __name__ == "__main__":
    a = tnp.random.rand(1_000_000)
    t1 = plain()
    t2 = tensor()
    print(f'Plain CPU: {t1}')
    print(f'Tensor: {t2}')
