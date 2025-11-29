import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow Version: {tf.__version__}")

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ SUCCESS: GPU Detected: {gpus}")
    # Verify we can actually run a calculation on the GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("✅ GPU Computation Test Passed")
            print(f"Result:\n{c}")
    except RuntimeError as e:
        print(f"❌ GPU Detected but computation failed: {e}")
else:
    print("❌ ERROR: No GPU found. TensorFlow is running on CPU.")

    # Debugging info if GPU is missing
    print("\nTroubleshooting Info:")
    import sys
    print(f"Python: {sys.version}")

    site_packages = [p for p in sys.path if 'site-packages' in p][0]
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    if os.path.exists(nvidia_dir):
        print(f"Nvidia Pip packages found at: {nvidia_dir}")
    else:
        print("Nvidia Pip packages NOT found)")
