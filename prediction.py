import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from keras.models import Sequential
from keras import layers
from keras.layers.experimental import preprocessing

dataset_dir = pathlib.Path("FlowersAi/flower_photos")
batch_size = 32
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = 'training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

num_classes = len(class_names)
model = Sequential([
    preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
	layers.experimental.preprocessing.RandomRotation(0.1),
	layers.experimental.preprocessing.RandomZoom(0.1),
	layers.experimental.preprocessing.RandomContrast(0.2),    
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer="adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.load_weights("FlowersAi/my_flowers_model.h5")

loss, acc = model.evaluate(train_ds, verbose=2)
print("Resored model, accuracy: {:5.2f}%".format(100*acc))

img = tf.keras.utils.load_img("FlowersAi/test.jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("На зображенні скоріш за все {} ({:.2f}% вірогідність)".format(
    class_names[np.argmax(score)],
    100*np.max(score))
)
