import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Number of GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Test GPU device name
print("GPU device name:", tf.test.gpu_device_name())