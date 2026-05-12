import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_PATH = r"D:\drive-federated"
GLOBAL_PATH = os.path.join(BASE_PATH, "global")
CLIENT_PATH = os.path.join(BASE_PATH, "client_updates")

NUM_CLIENTS = 2
ROUNDS = 10

# Model
def create_model():
    base = tf.keras.applications.EfficientNetB4(
        input_shape=(380, 380, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False  # transfer learning

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Wait for client files
def wait_for_clients(round_num):
    print(f"⏳ Waiting for clients (Round {round_num})...")
    while True:
        files = [
            f for f in os.listdir(CLIENT_PATH)
            if f.endswith(f"round_{round_num}.weights.h5")
        ]
        if len(files) >= NUM_CLIENTS:
            break
        time.sleep(3)


# Aggregation (FedAvg)
def aggregate(round_num):
    weights = []

    for i in range(NUM_CLIENTS):
        model = create_model()
        path = os.path.join(CLIENT_PATH, f"client_{i}_round_{round_num}.weights.h5")
        model.load_weights(path)
        weights.append(model.get_weights())

    avg_weights = []
    for layer in zip(*weights):
        avg_weights.append(np.mean(layer, axis=0))

    global_model = create_model()
    global_model.set_weights(avg_weights)

    return global_model


# MAIN
accuracy_history = []

# Create validation dataset ONCE (outside loop)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(380, 380),
    batch_size=32
)

# EfficientNet preprocessing
val_ds = val_ds.map(
    lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y)
)

# Initialize global model
global_model = create_model()
global_model.save(os.path.join(GLOBAL_PATH, "global_round_1.h5"))

for r in range(1, ROUNDS + 1):
    print(f"\n🚀 ROUND {r}")

    # Wait for clients
    wait_for_clients(r)

    # Aggregate
    global_model = aggregate(r)

    # Evaluate properly
    loss, acc = global_model.evaluate(val_ds, verbose=0)

    print(f"✅ Global Accuracy: {acc:.4f}")
    accuracy_history.append(acc)

    # Save next global model
    global_model.save(
        os.path.join(GLOBAL_PATH, f"global_round_{r+1}.h5")
    )

# Plot graph
plt.plot(range(1, ROUNDS+1), accuracy_history, marker='o')
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning Progress")
plt.grid()
plt.savefig(os.path.join(BASE_PATH, "accuracy_plot.png"))
plt.show()