
import cv2
import glob
import os

def change_files(old_ext, new_ext):
    path = 'dataset'
    files = glob.glob(path + '/**/*.' + old_ext, recursive=True)

    for file in files:
        img = cv2.imread(file)
        print(file)

        cv2.imwrite(file[:-len(old_ext)] + new_ext, img)
        os.remove(file)


def resize_files(ext, size):
    path = 'dataset/train'
    files = glob.glob(path + '/**/*.' + ext, recursive=True)

    for file in files:
        img = cv2.imread(file)
        print(file)
        img = cv2.resize(img, size)
        cv2.imwrite(file, img)



def main():

    old_ext = 'jpeg'
    new_ext = 'jpg'

    # change_files(old_ext, new_ext)
    resize_files(new_ext, (512, 512))


if __name__ == '__main__':
    main()