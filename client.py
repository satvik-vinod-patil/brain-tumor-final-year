import os
import time
import sys
import tensorflow as tf

client_id = int(sys.argv[1])

BASE_PATH = r"D:\drive-federated"
GLOBAL_PATH = os.path.join(BASE_PATH, "global")
CLIENT_PATH = os.path.join(BASE_PATH, "client_updates")

ROUNDS = 10

# Data preprocessing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    image_size=(380, 380),
    batch_size=32
)

# Augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y: (data_aug(x), y))


def create_model():
    base = tf.keras.applications.EfficientNetB4(
        input_shape=(380, 380, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

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


def wait_for_global(round_num):
    path = os.path.join(GLOBAL_PATH, f"global_round_{round_num}.h5")
    while not os.path.exists(path):
        time.sleep(3)
    return path


for r in range(1, ROUNDS + 1):
    print(f"📡 Client {client_id} - Round {r}")

    # Wait for global model
    global_model_path = wait_for_global(r)

    model = create_model()
    model.load_weights(global_model_path)

    # Train
    model.fit(train_ds, epochs=1, verbose=1)

    # Save update
    save_path = os.path.join(
    CLIENT_PATH,
    f"client_{client_id}_round_{r}.weights.h5"
    )
    model.save_weights(save_path)

    print(f"✅ Client {client_id} uploaded round {r}")