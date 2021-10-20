import cv2
import os
import sys

def write_grayscale(dir_path, out_path):
    os.mkdir(out_path)
    for filename in os.listdir(dir_path):
        if filename[-3:] == 'jpg':
            a = cv2.imread(dir_path+'/'+filename, 0)
            cv2.imwrite(out_path+'/gray_'+filename,a)
            cv2.waitKey(0)

if __name__ == "__main__":
    if not (len(sys.argv)==3):
        sys.exit()
    
    write_grayscale(sys.argv[1], sys.argv[2])
    sys.exit()