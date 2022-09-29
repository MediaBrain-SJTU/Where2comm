import os
import numpy as np
import cv2

# dair_path = {
#     'single': 'dair_single_2022_09_19_20_24_05_FromKris',
#     'late': 'dair_single_late_2022_09_19_20_24_05_FromKris',
#     'v2vnet': 'dair_npj_v2vnet_w_2022_09_17_19_50_39',
#     'disconet': 'dair_npj_pointpillar_disconet_w_2022_09_13_11_12_24',
#     'when2com': 'dair_npj_when2com_2022_09_18_21_30_04',
#     'where2comm': 'dair_npj_where2comm_multiscale_resnet_max_2022_09_14_23_08_17',
#     'v2xvit': 'dair_npj_v2xvit_w_2022_09_06_12_31_46_FromKris'
# }

dair_path = {
    'single': 'logs/dair_single_late_2022_09_22_20_19_14',
    'disconet': 'logs/dair_disconet_2022_09_22_20_27_45',
    'when2com': 'logs/dair_when2com_2022_09_22_23_07_02',
    'where2comm': 'bp_logs/dair_npj_where2comm_multiscale_resnet_max_2022_09_14_23_08_17'
}


v2x_path = {
    'single': 'logs/v2x2_single_2022_09_22_01_28_30',
    'late': 'logs/v2x2_single_late_2022_09_22_01_28_30',
    'where2comm': 'logs/v2x2_npj_where2comm_multiscale_resnet_max_2022_09_22_01_17_57'
}

# Crop and merge image
def load_img(root_dir, img_id, mode):
    img_dir = os.path.join('opencood/', root_dir, 'vis_{}'.format(mode), '{}_{:05d}.png'.format(mode, img_id))
    if not os.path.exists(img_dir):
        print(img_dir)
        return None
    img = cv2.imread(img_dir)
    H, W, _ = img.shape
    if mode == '3d':
        h, w = 340, 50
    else:
        h, w = 460, 50 # for dair
        # h, w = 50, 350  # for v2x
    img = img[h:H-h,w:W-w]
    return img

dataset = 'dair'
root_save_path = 'opencood/visualization/merged/{}'.format(dataset)
if not os.path.exists(root_save_path):
    os.mkdir(root_save_path)

def merge_img(img_id, picked_model, align='row', mode='3d'):
    images = []
    for model in picked_model:
        image = load_img(model, img_id, mode)
        if image is not None:
            images.append(image)
        else:
            return None
    
    if align == 'row':
        images = np.concatenate(images, axis=1)
    else:
        images = np.concatenate(images, axis=0)
    
    image_save_path = os.path.join(root_save_path, mode, '{:05d}.png'.format(img_id))
    print(image_save_path)
    # if not os.path.exists(image_save_path):
    cv2.imwrite(image_save_path, images)
    return images

# mode = '3d'
# align = 'row'
mode = 'bev'
align = 'col'
start_idx = 0
end_idx = 100
gap = 100

picked_mode = ['single', 'when2com', 'disconet', 'where2comm']
# dataset = 'v2x'
# picked_model = [v2x_path[x] for x in picked_mode]
picked_model = [dair_path[x] for x in picked_mode]
video_save_path = os.path.join(root_save_path, 'result_video')
if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)


for start_id in range(start_idx, end_idx, gap):
    end_id = start_id + gap
    for mode, align in zip(['bev', '3d'], ['col', 'row']):
        save_path = os.path.join(root_save_path, mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_array = []
        for img_id in range(start_id, end_id):
            image = merge_img(img_id, picked_model, align, mode)

            if image is None:
                continue
            height, width, layers = image.shape
            size = (width,height)
            img_array.append(image)

        out = cv2.VideoWriter(os.path.join(root_save_path, 'result_video/{}_{}_{}.mp4'.format(dataset, mode, start_id, end_id)), cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        break