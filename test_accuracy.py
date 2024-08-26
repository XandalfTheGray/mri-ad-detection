import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow import math as tfmath
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('ADnet_inception')

if len(sys.argv) < 2:
    path_to_photos = os.getcwd() + '/ADNI_Pictures'

else:
    path_to_photos = sys.argv[1] + '/ADNI_Pictures'

test_dir = os.path.join(path_to_photos, 'test')
test_dir_AD = os.path.join(test_dir, 'AD')
test_dir_CN = os.path.join(test_dir, 'CN')

num_AD = len(os.listdir(test_dir_AD))
num_CN = len(os.listdir(test_dir_CN))

test_image_gen = ImageDataGenerator(rescale=1/255.0)
batch_size = 32
dim1 = 256
dim2 = 256

test_data_gen = test_image_gen.flow_from_directory(batch_size=1,
                                                   directory=test_dir,
                                                   shuffle=False,
                                                   target_size=(dim1, dim2),
                                                   color_mode='rgb',
                                                   class_mode='binary')

numImgs = 2601
labels = test_data_gen.labels
predictions = model.predict(test_data_gen, batch_size=1, verbose=1, steps=numImgs)
tn = 0
tp = 0
fp = 0
fn = 0

for idx in range(0, len(predictions), 17):
    img_preds = predictions[idx:idx+17]
    img_label = np.around(np.average(img_preds))

    if img_label == 1 and labels[idx] == 1:
        tn += 1
    elif img_label == 0 and labels[idx] == 0:
        tp += 1
    elif img_label == 1 and labels[idx] == 0:
        fn += 1
    elif img_label == 0 and labels[idx] == 1:
        fp += 1
 
conf_mat_avg = [[tp, fn], [fp, tn]]
print(conf_mat_avg)
conf_mat = tfmath.confusion_matrix(labels, np.around(predictions[:numImgs]))
print(predictions)
print(conf_mat)
