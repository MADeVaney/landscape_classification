import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_ds = tf.keras.utils.image_dataset_from_directory(
  '../Aerial_Landscapes',
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(256, 256),
  batch_size=10)

val_ds = tf.keras.utils.image_dataset_from_directory(
  '../Aerial_Landscapes',
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(256, 256),
  batch_size=10)


#class_names = train_ds.class_names
#print(class_names)

#normalization_layer = tf.keras.layers.Rescaling(1./255)

'''for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break'''

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 15

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(8, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(rate=0.25), #tried in several different position with no noticeable differences
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=6
)

model.save('model.keras')
