import tensorflow as tf
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, Dense, Flatten, MaxPooling3D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import prep_data as prep
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras.constraints import max_norm

# Default image directories if not specified
if len(sys.argv) < 2:
    path_to_photos = os.getcwd() + '/ADNI_Pictures'
else:
    path_to_photos = sys.argv[1] + '/ADNI_Pictures'

# Locations of train, test, and val datasets
train_dir = os.path.join(path_to_photos, 'train')
val_dir = os.path.join(path_to_photos, 'validation')
test_dir = os.path.join(path_to_photos, 'test')
num_AD_tr = len(os.listdir(train_AD_dir))
num_CN_tr = len(os.listdir(train_CN_dir))
num_AD_val = len(os.listdir(val_AD_dir))
num_CN_val = len(os.listdir(val_CN_dir))

# Define batch size and training epochs
batch_size = 32
epochs = 50

# input dimensions
dim1 = 256
dim2 = 256
dim3 = 3

imgShape = (dim1, dim2, dim3)

init_model = tf.keras.applications.InceptionV3(include_top=False,
                                               weights='imagenet',
                                               input_shape=imgShape)

new_input = init_model.input
print(new_input)
init_model.trainable = True

# Random zoom in from 0% to 30% zoom
train_image_gen = ImageDataGenerator(rescale=1/255.0, zoom_range=[0.7, 1])
val_image_gen = ImageDataGenerator(rescale=1/255.0)
test_image_gen = ImageDataGenerator(rescale=1/255.0)

# Flow from directories to get data
train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(dim1, dim2),
                                                     color_mode='rgb',
                                                     class_mode='binary')

print(train_data_gen.class_indices)

val_data_gen = val_image_gen.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(dim1, dim2),
                                                 color_mode='rgb',
                                                 class_mode='binary')

test_data_gen = test_image_gen.flow_from_directory(batch_size=batch_size,
                                                   directory=test_dir,
                                                   target_size=(dim1, dim2),
                                                   color_mode='rgb',
                                                   class_mode='binary')

model = tf.keras.Sequential([
    init_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
 ])

print(model.summary())

# define early stopping parameters with patience 5
# Restore best weights when finished
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                      restore_best_weights=True)

# Define SGD and Adam optimizer parameters
optim = tf.keras.optimizers.SGD(learning_rate=0.01)
optim2 = tf.keras.optimizers.Adam(learning_rate=0.001)

# Use binary crossentropy loss function
model.compile(optimizer=optim2,
              loss='binary_crossentropy',
              metrics=['accuracy'])
              #class_weights = { 0:1.25, 1:0.83}

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=batch_size,
    callbacks=[es]
)

# Evaluate on test dataset
score = model.evaluate(test_data_gen, steps=100, batch_size=48)

# Print training progress to figures
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
idx = 0
fname = 'ADnetAccuracy'

while(os.path.exists(fname + str(idx) + '.png')):
    idx += 1

plt.savefig(fname + str(idx) + '.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('ADnetLoss' + str(idx) + '.png')
plt.close()

# Save the model
model.save('InceptionV3')