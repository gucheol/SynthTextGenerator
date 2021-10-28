import numpy as np
import math 
import PIL
import json
import cv2 
import os
import glob
import tqdm 

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
    width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))    
    height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    dst = np.array([[0, 0],[width - 1, 0],[width - 1, height - 1],[0, height - 1]], dtype = "float32")
    # print(rect)
    # print(dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped


def extract(src_path, target_path):
    sequence = 0 
    for json_file_path in tqdm.tqdm(glob.glob(src_path + '/json/*.json')):
        with open(json_file_path, 'r', encoding='utf8') as f:
            json_obj = json.load(f)
        rgb_file_name = json_obj['info_list'][0]["image_filename"]
        rgb_file_name = rgb_file_name.replace('seg', 'rgb')
        im = cv2.imread(os.path.join(src_path, 'rgb', rgb_file_name))
        for segment in json_obj['info_list'][0]['segment_list']:
            for word in segment['words']:
                transcription = word['transcription']
                points = np.array([[int(point['x']), int(point['y'])] for point in word['word_points']])
                # print(points)
                # im = cv2.polylines(im, [points], True, color=(255, 0, 0))
                # im = cv2.polylines(im, points, True, color=(255, 0, 0))
                
                # cv2.imshow('title', im)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite('all.jpg', im)

                patch_im = four_point_transform(im, np.float32(points))
                # cv2.imshow('patch', im)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(os.path.join(target_path, transcription + f'_{sequence}.jpg'))
                cv2.imwrite(os.path.join(target_path, transcription + f'_{sequence}.jpg'), patch_im)
                sequence += 1 



def extract_old():
    data_root_path = '/home/embian/Unity_Dataset/2021_01_12/'    
    out_path = './Unity_out'
    if not os.path.exists(out_path):
        os.mkdir(out_path) 

    for folder_name in os.listdir(data_root_path):        
        json_path = os.path.join(data_root_path, folder_name, 'BoundingBox.json')        
        if not os.path.exists(json_path):
            continue

        json_fp = open(json_path, 'r')
        json_entries = json.load(json_fp)

        if not os.path.exists(os.path.join(out_path, folder_name)) :
            os.mkdir(os.path.join(out_path, folder_name))

        for entries in json_entries['info_list']:
            seg_file_name = entries['image_filename']
            rgb_file_path = os.path.join(data_root_path, folder_name, 'rgb', seg_file_name.replace('seg_', 'rgb_'))

            if not os.path.exists(rgb_file_path) :
                print(f'{rgb_file_path} not found')
                continue

            im = cv2.imread(rgb_file_path)
            height, width = im.shape[:2]

            for mrz in entries['segment_list']:
                left_top = [int(mrz['quad']['left_top']['x']), 
                            int(mrz['quad']['left_top']['y'])]

                right_top = [int(mrz['quad']['right_top']['x']), 
                            int(mrz['quad']['right_top']['y'])]

                left_bottom = [int(mrz['quad']['left_bottom']['x']), 
                            int(mrz['quad']['left_bottom']['y'])]

                right_bottom= [int(mrz['quad']['right_bottom']['x']), 
                            int(mrz['quad']['right_bottom']['y'])]

                gt_text = mrz['text']
                gt_text = gt_text.replace('<', '-')

                card_bounds = np.array([left_top, right_top, right_bottom, left_bottom])
                
                
                if np.max(card_bounds[:, 0]) > width or np.max(card_bounds[:, 1]) > height or np.min(card_bounds) < 0 :
                    continue 

                rotation_angle = compute_rotation_angle(card_bounds)
                rotated_im, adjust_bounds = rotate_image(im, card_bounds[0], rotation_angle, card_bounds)   
                
                if adjust_bounds[2][1] - adjust_bounds[0][1] < 12 or adjust_bounds[2][0] - adjust_bounds[0][0] < 100 :
                    continue

                gt_image = rotated_im[adjust_bounds[0][1] : adjust_bounds[2][1], adjust_bounds[0][0] : adjust_bounds[2][0]]                
                out_file_path = os.path.join(out_path, folder_name, gt_text + '.jpg')
                try:
                    cv2.imwrite(out_file_path, gt_image, params=[cv2.IMWRITE_JPEG_QUALITY,100])
                except:
                    print (f'error : {out_file_path}, {rgb_file_path} {adjust_bounds[2][1] - adjust_bounds[0][1]} {adjust_bounds[2][0] - adjust_bounds[0][0]}')

def compute_rotation_angle(box_pos, theta_threshold=1):
    delta_x = box_pos[1][0] - box_pos[0][0]
    delta_y = box_pos[1][1] - box_pos[0][1]

    if delta_x == 0 and delta_y < 0:
        return -90
    elif delta_x == 0 and delta_y > 0:
        return 90

    theta = math.atan(delta_y/delta_x) * 180 / math.pi

    if abs(theta) < theta_threshold and delta_x > 0:
        return 0
    elif abs(theta) < theta_threshold and delta_x < 0:
        return -180

    if theta > 0 and delta_x > 0:
        theta = theta
    elif theta > 0 > delta_x:
        theta = theta - 180
    elif theta < 0 < delta_x:
        theta = theta
    else:
        theta = 180+theta

    return theta

def align_points(box):
    # Make it clockwise align
    # box = np.array([(int(float(x)), int(float(y))) for x, y in zip(items[1::2], items[2::2])])
    centroid = np.sum(box, axis=0) / 4
    theta = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0]) * 180 / np.pi
    indices = np.argsort(theta)
    aligned_box = box[indices]
    return aligned_box

def add_margin(box, width, height, margin=2):
    adjust_bounds = np.zeros_like(box)
    adjust_bounds[0][0] = max(box[0][0] - margin , 0) 
    adjust_bounds[0][1] = max(box[0][1] - margin , 0) 

    adjust_bounds[1][0] = min(box[1][0] + margin , width) 
    adjust_bounds[1][1] = max(box[1][1] - margin , 0) 

    adjust_bounds[2][0] = min(box[2][0] + margin , width) 
    adjust_bounds[2][1] = min(box[2][1] + margin , height) 

    adjust_bounds[3][0] = max(box[3][0] - margin , 0) 
    adjust_bounds[3][1] = min(box[3][1] + margin , height) 

    left = min(adjust_bounds[:, 0])
    right = max(adjust_bounds[:, 0])
    top_y = min(adjust_bounds[:, 1])
    bottom_y = max(adjust_bounds[:, 1])

    return np.array([[left, top_y], [right, top_y], [right, bottom_y], [left, bottom_y]])



def rotate_image(img, center, angle, bounds):
    if angle == 0:
        return img, add_margin(bounds, im.shape[1], im.shape[0])

    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    bounding_box = np.array([[0, 0], [0, height], [width, 0], [width, height]])
    adjust_box = np.transpose(np.dot(rotation_matrix[:, :2], np.transpose(bounding_box))) + rotation_matrix[:, 2]

    min_x = np.min(adjust_box[:, 0])
    min_y = np.min(adjust_box[:, 1])
    max_x = np.max(adjust_box[:, 0])
    max_y = np.max(adjust_box[:, 1])

    bound_w = int(max_x - min_x)
    bound_h = int(max_y - min_y)

    rotation_matrix[0, 2] -= min_x
    rotation_matrix[1, 2] -= min_y
    rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
    adjust_bounds = np.transpose(np.dot(rotation_matrix[:, :2], np.transpose(bounds))) + rotation_matrix[:, 2]

    adjust_bounds = align_points(adjust_bounds)
    adjust_height, adjust_width = rotated_img.shape[:2]
    adjust_bounds = add_margin(adjust_bounds, adjust_width, adjust_height)
    return rotated_img, np.int32(adjust_bounds)


if __name__=='__main__':
    src_path = '/home/embian/Unity_Dataset/2021_02_10/Alien/'
    target_path = './unity/Alien/'
    extract(src_path, target_path)
    
    src_path = '/home/embian/Unity_Dataset/2021_02_10/ID/'
    target_path = './unity/ID/'
    extract(src_path, target_path)

    src_path = '/home/embian/Unity_Dataset/2021_02_10/Driver/'
    target_path = './unity/Driver/'
    extract(src_path, target_path)

    src_path = '/home/embian/Unity_Dataset/2021_02_10/Overseas/'
    target_path = './unity/Overseas/'
    extract(src_path, target_path)

