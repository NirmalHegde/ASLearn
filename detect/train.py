import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

imgDirPath = 'Data-Collection/images'

batchSize = 5
imgHeight = 32
imgWidth = 32

train_ds = keras.preprocessing.image_dataset_from_directory(
    imgDirPath,
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z'],
    color_mode='grayscale',
    batch_size=batchSize,
    image_size=(imgHeight, imgWidth),
    shuffle=True,
    seed=100,
    validation_split=0.2,
    subset="training"
)

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomRotation(0.3),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.1)
])

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='sigmoid')
])

# print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy']
)

history = model.fit(train_ds, epochs=8)
model.save('anjalis-model.h5')
