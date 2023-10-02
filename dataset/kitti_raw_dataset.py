import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import argparse

MAX_DISP = 256
TEST_SEQUENCES = [
    '2011_09_30_drive_0028_sync'
]
CITY_SEQUENCES = [
    '2011_09_26_drive_0001_sync',
    '2011_09_26_drive_0002_sync',
    '2011_09_26_drive_0005_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0011_sync',
    '2011_09_26_drive_0013_sync',
    '2011_09_26_drive_0014_sync',
    '2011_09_26_drive_0017_sync',
    '2011_09_26_drive_0018_sync',
    '2011_09_26_drive_0048_sync',
    '2011_09_26_drive_0051_sync',
    '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0057_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0060_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0091_sync',
    '2011_09_26_drive_0093_sync',
    '2011_09_26_drive_0095_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0104_sync',
    '2011_09_26_drive_0106_sync',
    '2011_09_26_drive_0113_sync',
    '2011_09_26_drive_0117_sync',
    '2011_09_28_drive_0001_sync',
    '2011_09_28_drive_0002_sync',
    '2011_09_29_drive_0026_sync',
    '2011_09_29_drive_0071_sync',
]

RESIDENTIAL_SEQUENCES = [
    '2011_09_26_drive_0019_sync',
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0022_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0035_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0039_sync',
    '2011_09_26_drive_0046_sync',
    '2011_09_26_drive_0061_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0079_sync',
    '2011_09_26_drive_0086_sync',
    '2011_09_26_drive_0087_sync',
    '2011_09_30_drive_0018_sync',
    '2011_09_30_drive_0020_sync',
    '2011_09_30_drive_0027_sync',
    '2011_09_30_drive_0028_sync',
    '2011_09_30_drive_0033_sync',
    '2011_09_30_drive_0034_sync',
    '2011_10_03_drive_0027_sync',
    '2011_10_03_drive_0034_sync',
]

CAMPUS_SEQUENCES = [
    '2011_09_28_drive_0016_sync',
    '2011_09_28_drive_0021_sync',
    '2011_09_28_drive_0034_sync',
    '2011_09_28_drive_0035_sync',
    '2011_09_28_drive_0037_sync',
    '2011_09_28_drive_0038_sync',
    '2011_09_28_drive_0045_sync',
    '2011_09_28_drive_0047_sync',
]
ROAD_SEQUENCES = [
    '2011_09_26_drive_0015_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0028_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0032_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0070_sync',
    '2011_09_26_drive_0101_sync',
    '2011_09_29_drive_0004_sync',
    '2011_09_30_drive_0016_sync',
    '2011_10_03_drive_0042_sync',
    '2011_10_03_drive_0047_sync',
]

def get_sgm_disp(imageL, imageR,
            right_disp = False,
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
    if right_disp:
        # cv2.imshow("L", imageL)
        # cv2.imshow("R", imageR)
        flipped_imageL = cv2.flip(imageL,1)
        flipped_imageR = cv2.flip(imageR,1)
        imageL = flipped_imageR
        imageR = flipped_imageL
        # cv2.imshow("after L", imageL)
        # cv2.imshow("after R", imageR)
        # cv2.waitKey()
        # cv2.destroyWindow()
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


def get_kitti2017_datapath(kitti_dir, dir_name_list, gt_dir):
    """ Read path to all data from KITTI Depth Prediction dataset """
    left_data_path = {'rgb': [], 'depth': [], 'disp': [], 'proxy': []}
    right_data_path = {'rgb': [], 'depth': [], 'disp': [], 'proxy': []}

    for dir_name in dir_name_list:
        # Directory of RGB images
        rgb_left_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'image_02', 'data')
        rgb_right_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'image_03', 'data')
        # Directory of ground truth depth maps
        depth_left_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'proj_depth', 'groundtruth', 'image_02')
        depth_right_dir = os.path.join(kitti_dir, dir_name[:-16], dir_name, 'proj_depth', 'groundtruth', 'image_03')
        # Directory of ground truth disparity maps
        disp_left_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, 'proj_disp', 'image_02')
        disp_right_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, 'proj_disp','image_03')
        proxy_left_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, f'{args.proxy}_disp','image_02')
        proxy_right_dir = os.path.join(gt_dir, dir_name[:-16], dir_name, f'{args.proxy}_disp','image_03')


        if not os.path.exists(disp_left_dir):
            os.makedirs(disp_left_dir)
        if not os.path.exists(disp_right_dir):
            os.makedirs(disp_right_dir)
        if not os.path.exists(proxy_left_dir):
            os.makedirs(proxy_left_dir)
        if not os.path.exists(proxy_right_dir):
            os.makedirs(proxy_right_dir)


        # Get image names (DO NOT obtain from raw data directory since the annotated data is pruned)
        file_name_list = sorted(os.listdir(depth_left_dir))

        for filename in file_name_list:
            pfm_filename = filename.replace('.png','.pfm')

            # Path to RGB images
            rgb_left_path = os.path.join(rgb_left_dir, filename)
            rgb_right_path = os.path.join(rgb_right_dir, filename)
            # Path to ground truth depth maps
            depth_left_path = os.path.join(depth_left_dir, filename)
            depth_right_path = os.path.join(depth_right_dir, filename)
            # Path to ground truth disparity maps
            disp_left_path = os.path.join(disp_left_dir, filename)
            disp_right_path = os.path.join(disp_right_dir, filename)
            proxy_left_path = os.path.join(proxy_left_dir, filename)
            proxy_right_path = os.path.join(proxy_right_dir, filename)


            # Add to list
            left_data_path['rgb'].append(rgb_left_path)
            right_data_path['rgb'].append(rgb_right_path)
            left_data_path['depth'].append(depth_left_path)
            right_data_path['depth'].append(depth_right_path)
            left_data_path['disp'].append(disp_left_path)
            right_data_path['disp'].append(disp_right_path)
            left_data_path['proxy'].append(proxy_left_path)
            right_data_path['proxy'].append(proxy_right_path)

    return left_data_path, right_data_path

def main(args):
    # File system
    root_dir = args.input
    kitti_raw_dir = os.path.join(root_dir, 'kitti')
    gt_dir = os.path.join(args.output, 'groundtruth')
    csv_dir = os.path.join(args.output, 'path_list')

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    step = 0
    # KITTI RAW data
    keys = {'city':CITY_SEQUENCES, 'residental':RESIDENTIAL_SEQUENCES, 'campus':CAMPUS_SEQUENCES, 'road':ROAD_SEQUENCES, 'test':TEST_SEQUENCES}

    # Get data path
    left_data_path, right_data_path = get_kitti2017_datapath(kitti_raw_dir, keys[args.type], gt_dir)
    print(len(left_data_path['rgb']), len(left_data_path['depth']), len(left_data_path['disp']), len(left_data_path['proxy']))
    # Get ground truth disparity map
    for idx in range(len(left_data_path['depth'])):
        if args.save_disp:
            # Get ground truth disparity map
            depth_l = read_depth(left_data_path['depth'][idx])
            disp_l = depth2disp(depth_l, left_data_path['depth'][idx][-73:-63])
            # depth_r = read_depth(right_data_path['depth'][idx])
            # disp_r = depth2disp(depth_r, right_data_path['depth'][idx][-73:-63])

            # Get rgb image
            img_name_l = left_data_path['rgb'][idx]
            img_name_r = right_data_path['rgb'][idx]
            img_l = cv2.imread(img_name_l)
            img_r = cv2.imread(img_name_r)

            # Save disparity map
            disp_l_to_save = np.clip(disp_l, 0, MAX_DISP)
            disp_l_to_save = (disp_l_to_save).astype(np.float32)
            # disp_r_to_save = np.clip(disp_r, 0, MAX_DISP)
            # disp_r_to_save = (disp_r_to_save).astype(np.float32)
            cv2.imwrite(left_data_path['disp'][idx], disp_l_to_save)
            # cv2.imwrite(right_data_path['disp'][idx], disp_r_to_save)

            # Save proxy disparity map
            if args.proxy != 'None':
                proxy_l = get_sgm_disp(img_l, img_r)
                cv2.imwrite(left_data_path['proxy'][idx], proxy_l)
                if args.save_right:
                    proxy_r = get_sgm_disp(img_l, img_r, right_disp=True)
                    cv2.imwrite(right_data_path['proxy'][idx], cv2.flip(proxy_r,1))

            if step % 100 == 0:
                print('{}/{}'.format(step,len(left_data_path['disp'])))
            step+=1

    if args.save_csv:
        #TODO: right_disp ㅈㅓㅈㅏㅇ
        if args.proxy != 'None':
            if args.save_right:
                df = pd.DataFrame(
                    {'left_rgb': left_data_path['rgb'],
                     'right_rgb': right_data_path['rgb'],
                     'left_disp': left_data_path['disp'],
                     'left_proxy': left_data_path['proxy'],
                     'right_proxy': right_data_path['proxy']
                     })
                df.to_csv(os.path.join(csv_dir, f'kitti_{args.type}_{args.proxy}_fusion.csv'), index=False, header=False)
            else:
                df = pd.DataFrame(
                    {'left_rgb': left_data_path['rgb'],
                     'right_rgb': right_data_path['rgb'],
                     'left_disp': left_data_path['disp'],
                     'proxy_disp': left_data_path['proxy']})
                df.to_csv(os.path.join(csv_dir, f'kitti_{args.type}_{args.proxy}.csv'), index= False, header=False)
        else:
            df = pd.DataFrame(
                {'left_rgb': left_data_path['rgb'],
                 'right_rgb': right_data_path['rgb'],
                 'left_disp': left_data_path['disp']})
            df.to_csv(os.path.join(csv_dir, f'kitti_{args.type}.csv'), index= False, header=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for get data path file and kitti ground truth disparity map')
    parser.add_argument("-i", "--input", help='root path to the kitti raw file', default='/home/cvlab/nfs_clientshare/datasets')
    parser.add_argument("-o", "--output", help="path to the output folder where the results will be saved", default='/home/cvlab/PycharmProjects/continual_adaptation_deep_stereo/')
    parser.add_argument("--type", help="type of dataset: city, residential, campus, road", required=True)
    parser.add_argument("--save_disp", help="save disparities or not", type=int, default=1)
    parser.add_argument("--proxy", help="save proxies: None, proxy, sgm", default='None')
    parser.add_argument("--save_csv", help="save path file to csv or not", type=int, default=1)
    parser.add_argument("--save_right", type=int, default=0)
    # TODO: right disparity argument

    args = parser.parse_args()

    main(args)


