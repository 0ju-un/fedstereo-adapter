import tensorflow as tf
import numpy as np
import cv2
import re
import os
from collections import defaultdict

from Data_utils import preprocessing


def readPFM(file):
    """
    Load a pfm file as a numpy array
    Args:
        file: path to the file to be loaded
    Returns:
        content of the file as a numpy array
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dims = file.readline()
    try:
        width, height = list(map(int, dims.split()))
    except:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_list_file(path_file):
    """
    Read dataset description file encoded as left;right;disp;conf
    Args:
        path_file: path to the file encoding the database
    Returns:
        [left,right,gt,conf] 4 list containing the images to be loaded
    """
    with open(path_file,'r') as f_in:
        lines = f_in.readlines()
    lines = [x for x in lines if not x.strip()[0] == '#']
    left_file_list = []
    right_file_list = []
    gt_file_list = []
    l_proxy_file_list = []
    r_proxy_file_list = []
    r_proxy = False
    for l in lines:
        to_load = re.split(',|;',l.strip())
        left_file_list.append(to_load[0])
        right_file_list.append(to_load[1])
        if len(to_load)>2:
            gt_file_list.append(to_load[2])
        if len(to_load)>3:
            l_proxy_file_list.append(to_load[3])
        if len(to_load)>4:
            r_proxy=True
            r_proxy_file_list.append(to_load[4])
    if r_proxy:
        return left_file_list, right_file_list, gt_file_list, l_proxy_file_list, r_proxy_file_list

    return left_file_list,right_file_list,gt_file_list,l_proxy_file_list

def read_image_from_disc(image_path,shape=None,dtype=tf.uint8):
    """
    Create a queue to hoold the paths of files to be loaded, then create meta op to read and decode image
    Args:
        image_path: metaop with path of the image to be loaded
        shape: optional shape for the image
    Returns:
        meta_op with image_data
    """
    print('---------------------------------------------')
    # tf.Session().run([tf.global_variables_initializer()])
    image_raw = tf.read_file(image_path)
    if dtype==tf.uint8:
        image = tf.image.decode_image(image_raw)
    else:
        image = tf.image.decode_png(image_raw,dtype=dtype)
    if shape is None:
        image.set_shape([None,None,3])
    else:
        image.set_shape(shape)
    return tf.cast(image, dtype=tf.float32)


class dataset():
    """
    Class that reads a dataset for deep stereo
    """
    def __init__(
        self,
        path_file,
        batch_size=4,
        crop_shape=[320,1216],
        num_epochs=None,
        augment=False,
        is_training=True,
        proxies=False,
        shuffle=True,
        r_proxy=False):

        if not os.path.exists(path_file):
            raise Exception('File not found during dataset construction')
    
        self._path_file = path_file
        self._batch_size=batch_size
        self._crop_shape=crop_shape
        self._num_epochs=num_epochs
        self._augment=augment
        self._shuffle=shuffle
        self._is_training = is_training
        self._r_proxy = r_proxy
        self._build_input_pipeline_with_proxies()


    def _load_image_with_proxies(self, files):
        print(files)
        left_file_name = files[0]
        right_file_name = files[1]
        gt_file_name = files[2]
        l_proxy_file_name = files[3]
        if self._r_proxy:
            r_proxy_file_name = files[4]

        left_image = read_image_from_disc(left_file_name)
        right_image = read_image_from_disc(right_file_name)
        if self._usePfm:
            gt_image = tf.py_func(lambda x: readPFM(x)[0], [gt_file_name], tf.float32)
            gt_image.set_shape([None,None,1])
        else:
            read_type = tf.uint16 if self._double_prec_gt else tf.uint8
            gt_image = read_image_from_disc(gt_file_name,shape=[None,None,1], dtype=read_type)
            gt_image = tf.cast(gt_image,tf.float32)
            if self._double_prec_gt:
                gt_image = gt_image/256.0

        gt_image =  gt_image[:,:tf.shape(left_image)[1],:]

        px_read_type = tf.uint16 if self._double_prec_px else tf.uint8
        l_proxy_image = read_image_from_disc(l_proxy_file_name,shape=[None,None,1], dtype=px_read_type)
        l_proxy_image = tf.cast(l_proxy_image,tf.float32)
        #TODO: right proxy ㄱㅗㅏㄴㄹㅕㄴ ㅋㄷㅡ ㄱㅗㅊㄱㅣ
        if self._r_proxy:
            r_proxy_image = read_image_from_disc(r_proxy_file_name, shape=[None, None, 1], dtype=px_read_type)
            r_proxy_image = tf.cast(r_proxy_image, tf.float32)
            r_proxy_image = r_proxy_image[:, :tf.shape(left_image)[1], :]
            if self._double_prec_px:
                r_proxy_image = r_proxy_image / 256.0

        if self._double_prec_px:
                l_proxy_image = l_proxy_image/256.0

        real_width = tf.shape(left_image)[1]

        l_proxy_image = l_proxy_image[:,:tf.shape(left_image)[1],:]
        if self._r_proxy:
            (left_image, right_image, gt_image, l_proxy_image, r_proxy_image) = [
            tf.image.resize_image_with_crop_or_pad(x, self._crop_shape[0], self._crop_shape[1]) for x in
            [left_image, right_image, gt_image, l_proxy_image, r_proxy_image]]
        elif self._is_training:
            left_image,right_image,gt_image = preprocessing.random_crop(self._crop_shape, [left_image,right_image,gt_image])
        # elif self._r_proxy:
        #     (left_image,right_image,gt_image,l_proxy_image,r_proxy_image) = [tf.image.resize_image_with_crop_or_pad(x,self._crop_shape[0],self._crop_shape[1]) for x in [left_image,right_image,gt_image,l_proxy_image,r_proxy_image]]
        else:
            (left_image,right_image,gt_image,l_proxy_image) = [tf.image.resize_image_with_crop_or_pad(x,self._crop_shape[0],self._crop_shape[1]) for x in [left_image,right_image,gt_image,l_proxy_image]]

        if self._augment:
            left_image,right_image=preprocessing.augment(left_image,right_image)
        if self._r_proxy:
            print('rproxy')
            return [left_image, right_image, gt_image, l_proxy_image, r_proxy_image, real_width]
        return [left_image,right_image,gt_image,l_proxy_image,real_width]

    def _build_input_pipeline_with_proxies(self):
        if self._r_proxy:
            left_files, right_files, gt_files, l_proxy_files, r_proxy_files = read_list_file(self._path_file)
            self._couples = [[l, r, gt, lpx, rpx] for l, r, gt, lpx, rpx in zip(left_files, right_files, gt_files, l_proxy_files, r_proxy_files)]
        else:
            left_files, right_files, gt_files, l_proxy_files = read_list_file(self._path_file)
            self._couples = [[l, r, gt, lpx] for l, r, gt, lpx in zip(left_files, right_files, gt_files, l_proxy_files)]
        #flags
        self._usePfm = gt_files[0].endswith('pfm') or gt_files[0].endswith('PFM')
        if not self._usePfm:
            gg = cv2.imread(gt_files[0],-1)
            self._double_prec_gt = (gg.dtype == np.uint16)

        gg = cv2.imread(l_proxy_files[0],-1)
        self._double_prec_px = (gg.dtype == np.uint16)

        #create dataset
        dataset = tf.data.Dataset.from_tensor_slices(self._couples).repeat(self._num_epochs)
        # dataset = tf.data.Dataset.from_tensor_slices(self._couples)

        if self._shuffle:
            dataset = dataset.shuffle(self._batch_size*50)
        
        #load images
        dataset = dataset.map(self._load_image_with_proxies)

        #transform data
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=30)

        #get iterator and batches
        iterator = dataset.make_one_shot_iterator()
        images = iterator.get_next()
        print(images)
        self._left_batch = images[0]
        self._right_batch = images[1]
        self._gt_batch = images[2]
        self._l_px_batch = images[3]
        if self._r_proxy:
            self._r_px_batch = images[4]
            self._real_width = images[5]
        else:
            self._real_width = images[4]


    ################# PUBLIC METHOD #######################

    def __len__(self):
        return len(self._couples)
    
    def get_max_steps(self):
        return (len(self)*self._num_epochs)//self._batch_size

    def get_batch(self):
        if self._r_proxy:
            return self._left_batch, self._right_batch, self._gt_batch, self._l_px_batch, self._r_px_batch, self._real_width
        return self._left_batch,self._right_batch,self._gt_batch,self._l_px_batch,self._real_width
    
    def get_couples(self):
        return self._couples