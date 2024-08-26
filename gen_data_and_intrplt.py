import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys, re
import cv2

def getFileNames(folder): 
    filenames = []

    for root, dirs, files in os.walk(folder): 
        for fname in dirs + files:
            name = os.path.join(root, fname)
            
            if re.search(r".nii", name):
                filenames.append(name) 
                
    return filenames

def getDataLabels(filenames, lab0, lab1): 
    labels = []
    
    for fname in filenames:
        if re.search(lab0, fname): 
            labels.append(0)

        elif re.search(lab1, fname): 
            labels.append(1)

    #	print(labels) 
    return labels

def resizeTo(img, dim1, dim2): 
    x = len(img)
    y = len(img[0])
    padx = (dim1 - x)//2 
    pady = (dim2 - y)//2
    img2 = cv2.copyMakeBorder(img, padx, padx, pady, pady, cv2.BORDER_CONSTANT, value=[0])
    return img2

def saveToPNG(filename, data): 
    dim1 = len(data)
    midPoint = round(dim1/2)
    
    for shift in range(-16, 17, 2): 
        imgArray = data[midPoint+shift]
        fname = filename[0:-4] + "_shift" + str(shift) + ".png" 
        #img = cv2.resize(imgArray, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        img = resizeTo(imgArray, 256, 256)
        #print("X: " + str(len(img)) + "	Y: " + str(len(img[0]))) #print(fname)
        #print("xdim: " + str(len(img)) + "	ydim: " + str(len(img[midPoint]))) plt.imsave(fname, imgArray)
        return

def genPNGimages(dirpath):
    data_path = dirpath + "/ADNI_Pictures" 
    train_path = data_path + "/train" 
    val_path = data_path + "/validation"

    train_path_AD = train_path + "/AD" 
    train_path_CN = train_path + "/CN"
    
    val_path_AD = val_path + "/AD" 
    val_path_CN = val_path + "/CN"

    train_files = getFileNames(train_path)
    train_labels = getDataLabels(train_files, r"[\\/]AD[\\/]", r"[\\/]CN[\\/]") 
    val_files = getFileNames(val_path)
    val_labels = getDataLabels(val_files, r"[\\/]AD[\\/]", r"[\\/]CN[\\/]") 
    #print(train_files)
    #print(train_labels) 
    print(len(train_files)) 
    print(len(train_labels))


    for idx in range(len(train_labels)): 
        fname = train_files[idx]
        label = train_labels[idx] 
        data = nib.load(fname)
        data_np = data.get_data().astype('int16') 
        saveToPNG(fname, data_np)
        #print("x: " + str(len(data_np)) + "	y: " + str(len(data_np[0])) + " z: " + str(len(data_np[0][0])))


    for idx in range(len(val_labels)): 
        fname = val_files[idx]
        label = val_labels[idx] 
        data = nib.load(fname)
        data_np = data.get_data().astype('int16') 
        saveToPNG(fname, data_np)

