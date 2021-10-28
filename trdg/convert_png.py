import cv2
import glob

for file_path in glob.glob('./trdg/pictures/*.png'):
    target_path = file_path.replace('.png', '.jpg')
    im = cv2.imread(file_path)
    cv2.imwrite(target_path, im)
