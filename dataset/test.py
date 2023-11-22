from PIL import Image
import numpy as np
import cv2
import os

def read_depth(path):
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float) / 256.0 # pixel to meter
    return depth#[:, :, np.newaxis]

def main():
    proj_l_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/proj_disp/image_02'
    proj_r_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/proj_disp/image_03'
    sgm_l_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/sgm_disp/image_02'
    sgm_r_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/sgm_disp/image_03'
    proxy_l_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/proxy_disp/image_02'
    proxy_r_disp_dir = '/home/cvlab/PycharmProjects/continual_stereo/groundtruth/2011_09_26/2011_09_26_drive_0001_sync/proxy_disp/image_03'

    img_file = '0000000015.png'

    proj_l_disp_path = os.path.join(proj_l_disp_dir,img_file)
    proj_r_disp_path = os.path.join(proj_r_disp_dir,img_file)
    sgm_l_disp_path = os.path.join(sgm_l_disp_dir,img_file)
    sgm_r_disp_path = os.path.join(sgm_r_disp_dir,img_file)
    proxy_l_disp_path = os.path.join(proxy_l_disp_dir,img_file)
    proxy_r_disp_path = os.path.join(proxy_r_disp_dir,img_file)

    proj_l_disp = read_depth(proj_l_disp_path)
    proj_r_disp = read_depth(proj_r_disp_path)
    sgm_l_disp = read_depth(sgm_l_disp_path)
    sgm_r_disp = read_depth(sgm_r_disp_path)
    proxy_l_disp = read_depth(proxy_l_disp_path)
    proxy_r_disp = read_depth(proxy_r_disp_path)

    print()


if __name__ == '__main__':

    main()