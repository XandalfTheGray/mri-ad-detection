import os, sys

def get_imgs(path, pic_dest):
   
    entries = os.listdir(path) 
   
    if path == []:
        return

    for entry in entries:
        if os.path.isdir(path + '/' + entry):
            nextPath = path + '/' + entry 
            get_imgs(nextPath, pic_dest)

        elif entry.lower().endswith('.nii'):
            moveTo = pic_dest + '/' + entry 
            moveFrom = path + '/' + entry 
            os.rename(moveFrom, moveTo)

    return

if __name__ == "__main__":
    dirpath = os.getcwd() 
    pic_dest = dirpath + "/ADNI_Pictures" 
    if not (os.path.exists(pic_dest)):
        os.mkdir(pic_dest) 
    get_imgs(dirpath, pic_dest)
