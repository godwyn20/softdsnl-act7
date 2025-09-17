import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# 2. Pad sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# 3. Build model
model = models.Sequential([
    layers.Embedding(10000, 32, input_length=200),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train (store history)
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 6. Save model
model.save("imdb_text_model.h5")
print("✅ Model saved as imdb_text_model.h5")

# 7. Convert history to DataFrame
history_df = pd.DataFrame(history.history)

# 8. Create table image
fig, ax = plt.subplots(figsize=(8, 2))  # adjust size
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=history_df.round(4).values,
    colLabels=history_df.columns,
    rowLabels=[f"Epoch {i+1}" for i in range(len(history_df))],
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.savefig("training_history_table.png", bbox_inches="tight")
print("✅ Training history table saved as training_history_table.png")
