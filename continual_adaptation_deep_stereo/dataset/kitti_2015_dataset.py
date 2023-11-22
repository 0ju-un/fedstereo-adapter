import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import argparse

MAX_DISP = 256

def get_sgm_disp(imageL, imageR,
            min_disparity=0,
            num_disparities=MAX_DISP,
            block_size=5,
            window_size=5,
            disp12_max_diff=1,
            uniqueness_ratio=15,
            speckle_window_size=0,
            speckle_range=2,
            pre_filter_cap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
    # SGBM Parameters
    # http://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/?answer=183650#post-id-183650
    # window_size: default 3; 5 Works nicely
    #              7 for SGBM reduced size image;
    #              15 for SGBM full size image (1300px and above)
    # num_disparity: max_disp has to be dividable by 16 f. E. HH 192, 256

    p1 = 8 * 3 * window_size ** 2
    p2 = 32 * 3 * window_size ** 2
    param = {
        'minDisparity': min_disparity,
        'numDisparities': num_disparities,
        'blockSize': block_size,
        'P1': p1,
        'P2': p2,
        'disp12MaxDiff': disp12_max_diff,
        'uniquenessRatio': uniqueness_ratio,
        'speckleWindowSize': speckle_window_size,
        'speckleRange': speckle_range,
        'preFilterCap': pre_filter_cap,
        'mode': mode
    }
    stereoProcessor = cv2.StereoSGBM_create(**param)
    disparity = stereoProcessor.compute(imageL, imageR)
    disparity = disparity.astype(np.float32) / 16.0
    # normalize
    disp_img = disparity.copy()
    cv2.normalize(
        src=disp_img,
        dst=disp_img,
        beta=0,
        alpha=255,
        norm_type=cv2.NORM_MINMAX
    )
    return np.uint8(disp_img)

def depth2disp(depth, focal_key):
    """ Convert depth to disparity for KITTI dataset.
        NOTE: depth must be the original rectified images.
        Ref: https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py """
    baseline = 0.54
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
    width_to_focal[1238] = 718.3351

    focal_map = {
    '2011_09_26':7.215377e+02,
    '2011_09_28':7.070493e+02,
    '2011_09_29':7.183351e+02,
    '2011_09_30':7.070912e+02,
    '2011_10_03':7.188560e+02
    }

    baseline = 0.54
    focal_length = width_to_focal[depth.shape[1]]
    focal_len = focal_map[focal_key]
    invalid_mask = depth <= 0
    # disp = baseline * focal_length / (depth + 1E-8)
    disp = focal_len * baseline / (depth + 1E-8)
    # disp.inf -> 0
    disp[invalid_mask] = 0

    return disp

def read_depth(path):
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float) / 256.0 # pixel -> m
    return depth#[:, :, np.newaxis]


def get_kitti2015_datapath(kitti_dir, dir_name_list, gt_dir):
    """ Read path to all data from KITTI Depth Prediction dataset """
    data_path = {'imgL': [],'imgR': [], 'depth': [], 'gt_disp': [], 'proxy': [], 'filename': []}
    for dir_name in dir_name_list:
        # Directory of RGB images
        rgb_left_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'image_02', 'data')
        rgb_right_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'image_03', 'data')
        # Directory of ground truth depth maps
        depth_left_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'proj_depth', '../groundtruth', 'image_02')
        # Directory of ground truth disparity maps
        disp_left_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, 'proj_disp', '../groundtruth')
        proxy_disp_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, 'proxy_disp', '../groundtruth')

        if not os.path.exists(disp_left_dir):
            os.makedirs(disp_left_dir)

        if not os.path.exists(proxy_disp_dir):
            os.makedirs(proxy_disp_dir)

        # Get image names (DO NOT obtain from raw data directory since the annotated data is pruned)
        file_name_list = sorted(os.listdir(depth_left_dir))

        for filename in file_name_list:
            pfm_filename = filename.replace('.png','.pfm')
            # Path to RGB images
            rgb_left_path = os.path.join(rgb_left_dir, filename)
            rgb_right_path = os.path.join(rgb_right_dir, filename)
            # Path to ground truth depth maps
            depth_left_path = os.path.join(depth_left_dir, filename)
            # Path to ground truth disparity maps
            disp_left_path = os.path.join(disp_left_dir, filename)
            proxy_disp_path = os.path.join(proxy_disp_dir, filename)

            #.pfm
            # disp_left_path = os.path.join(disp_left_dir, pfm_filename)
            # proxy_disp_path = os.path.join(proxy_disp_dir, pfm_filename)

            # disp_left_path = os.path.join(disp_left_dir, filename)

            # Add to list
            data_path['imgL'].append(rgb_left_path)
            data_path['imgR'].append(rgb_right_path)
            data_path['depth'].append(depth_left_path)
            data_path['gt_disp'].append(disp_left_path)
            data_path['proxy'].append(proxy_disp_path)
            data_path['filename'].append(filename)



    return data_path

def main(args):
    # File system
    root_dir = args.input
    kitti_raw_dir = os.path.join(root_dir, 'kitti')
    gt_dir = os.path.join(args.output, '../groundtruth')
    csv_dir = os.path.join(args.output, '../path_list')

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    step = 0
    # KITTI RAW data
    keys = {'city':CITY_SEQUENCES, 'residental':RESIDENTIAL_SEQUENCES, 'campus':CAMPUS_SEQUENCES, 'road':ROAD_SEQUENCES, 'test':TEST_SEQUENCES}
    # Get data path
    data_path = get_kitti2017_datapath(kitti_raw_dir, keys[args.type], gt_dir)
    print(len(data_path['imgL']), len(data_path['depth']), len(data_path['gt_disp']), len(data_path['proxy']))
    # Get ground truth disparity map
    for idx, depth_path in enumerate(data_path['depth']):
        if args.save_disp:
            # Get ground truth disparity map
            gt_depth = read_depth(depth_path)
            gt_disp = depth2disp(gt_depth, depth_path[-73:-63])
            # Get proxy disparity map
            img_name_l = data_path['imgL'][idx]
            img_name_r = data_path['imgR'][idx]

            img_l = cv2.imread(img_name_l)
            img_r = cv2.imread(img_name_r)


            if step % 100 == 0:
                print('{}/{}'.format(step,len(data_path['gt_disp'])))

            # Save disparity map
            gt_dispy_to_save = np.clip(gt_disp, 0, MAX_DISP)
            gt_dispy_to_save = (gt_dispy_to_save).astype(np.float32)
            # dispy_to_save = dispy_to_save.astype(np.uint16)
            # dispy_to_save = (dispy_to_save*256.0).astype(np.uint16)
            cv2.imwrite(data_path['gt_disp'][idx], gt_dispy_to_save)
            # Save proxy disparity map
            if args.save_proxy:
                proxy_disp = get_sgm_disp(img_l, img_r)
                cv2.imwrite(data_path['proxy'][idx], proxy_disp)

            step+=1

    if args.save_csv:
        df_filename = pd.DataFrame({'files':data_path['filename']})
        df_filename.to_csv(os.path.join(csv_dir, f'kitti_{args.type}_files.csv'), index=False)
        if args.save_proxy:
            df = pd.DataFrame(
                {'left_rgb': data_path['imgL'], 'right_rgb': data_path['imgR'], 'gt_disp': data_path['gt_disp'], 'proxy': data_path['proxy']})
            df.to_csv(os.path.join(csv_dir, f'kitti_{args.type}_proxy.csv'), index=False)
        else:
            df = pd.DataFrame(
                {'left_rgb': data_path['imgL'], 'right_rgb': data_path['imgR'], 'gt_disp': data_path['gt_disp']})
            df.to_csv(os.path.join(csv_dir, f'kitti_{args.type}.csv'), index= False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for get data path file and kitti ground truth disparity map')
    parser.add_argument("-i", "--input", help='root path to the kitti raw file', default='/home/cvlab/nfs_clientshare/datasets')
    parser.add_argument("-o", "--output", help="path to the output folder where the results will be saved", default='/home/cvlab/PycharmProjects/continual_adaptation_deep_stereo/')
    parser.add_argument("--type", help="type of dataset: city, residential, campus, road", required=True)
    parser.add_argument("--save_disp", help="save disparities or not", type=int, default=1)
    parser.add_argument("--save_proxy", help="save disparities or not", type=int, default=1)
    parser.add_argument("--save_csv", help="save path file to csv or not", type=int, default=1)

    args = parser.parse_args()

    main(args)


