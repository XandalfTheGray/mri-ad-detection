import tensorflow as tf 
import os
import nibabel as nib 
import numpy as np

# x = iter(iterable)
# y = x.get_next().numpy() # z = y.decode("utf-8")
# data = nib.load(z)
# data_array = data.get_data() 

CLASSNAMES = np.array(['AD', 'CN'])

def getLabel(filename):
    fpath = tf.strings.split(filename, os.path.sep) 
    return fpath[-2] == CLASSNAMES

def getDataArray(filepath):
    img = nib.load(filepath).get_data() 
    return img

def process_path(filepath): 
    print(filepath)
    label = getLabel(filepath)
    fname = tf.io.read_file(filepath) 
    print(fname)
    img = getDataArray(fname)
    return img, label

