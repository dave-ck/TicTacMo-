import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution is: {}".format(tf.executing_eagerly()))
print("Keras version: {}".format(tf.keras.__version__))

var = tf.Variable([3, 3])

if tf.test.is_gpu_available():
    print('Running on GPU')
    print('GPU #0?')
    print(var.device.endswith('GPU:0'))
    print(var.device)
else:
    print('Running on CPU')


