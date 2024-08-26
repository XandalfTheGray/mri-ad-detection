import csv 
import os 
import re

def sort_imgs(path):
    filename = 'ADNI1_Complete_1Yr_1.5T_10_12_2021.csv' 
    files = []

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',') 
        
        for row in readCSV:
            files.append(row) 
            sub_ID = 1
            group = 2
            #path = os.getcwd()
            photo_fold = path + '/ADNI_Pictures' 
            
            if not os.path.exists(photo_fold):
                os.mkdir(photo_fold) 
            
            # Places all the MCI 
            MCI_fold = photo_fold + '/MCI'
            train_fold = photo_fold + '/train' 
            val_fold = photo_fold + '/validation' 
            train_fold_AD = train_fold + '/AD' 
            train_fold_CN = train_fold + '/CN' 
            val_fold_AD = val_fold + '/AD' 
            val_fold_CN = val_fold + '/CN'
            folds = [photo_fold, MCI_fold, train_fold, val_fold, 
                     train_fold_AD, train_fold_CN, val_fold_AD, val_fold_CN]

            for item in folds:
                if not os.path.exists(item): 
                    os.mkdir(item)
                    
            imageDir = path + '/ADNI_Pictures'
 
            pic_num = 0
            
            for line in files[1:]: 
                pic_num += 1
                subject = line[sub_ID]
                #regegg = "[]*(" + subject + "){1}[]*" label = line[group]
                #print(subject)
                images = os.listdir(imageDir)
                
                for image in images:
                    if not os.path.isdir(imageDir + '/' + image): 
                        isMatch = re.search(subject, image)
                        
                        if isMatch != None:
                            if pic_num % 10 == 0: 
                                data_set = '/validation/'

                            else:
                                data_set = '/train/'
                                fname = imageDir + '/' + image 
                                
                                if label != 'MCI':
                                    newName = imageDir + data_set + label + '/' + image
                                    
                                else:
                                    newName = imageDir + '/' + label + '/' + image 
                                
                                #print(label)
                                os.rename(fname, newName)