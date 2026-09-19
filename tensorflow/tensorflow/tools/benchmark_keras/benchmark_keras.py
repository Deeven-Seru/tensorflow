import tensorflow as tf
import time

def benchmark_keras_embedding():
    # Setup
    vocab_size = 1000000
    embed_dim = 128
    batch_size = 1024
    seq_len = 100

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Dummy data
    x = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    y = tf.random.uniform((batch_size, 1), dtype=tf.float32)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("Warming up...")
    # Warmup
    for _ in range(10):
        train_step(x, y)

    print("Running benchmark...")
    # Benchmark
    start = time.time()
    iterations = 50
    for _ in range(iterations):
        train_step(x, y)
    end = time.time()
    
    print(f"Keras Embedding Training - {iterations} iterations: {end - start:.4f} seconds")
    print(f"Time per step: {(end - start) / iterations * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark_keras_embedding()
