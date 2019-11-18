import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tf.debugging.set_log_device_placement(True)


print("Set environment variable for CUDA devices, and device debug to True.")

tf.constant(3)

# Place tensors on the CPU
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)
