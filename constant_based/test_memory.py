import numpy as np
try:
	np.zeros((512, 12, 4, 128, 128), dtype=np.float32)  # may raise MemoryError
except MemoryError:
	print("MemoryError: Unable to allocate array with shape (512, 12, 4, 128, 128) and dtype float32")
