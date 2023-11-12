import tensorflow as tf
import numpy as np
import cv2
import re
import os
from collections import defaultdict

from Data_utils import preprocessing
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

#TODO: temp code for print (remove later)
def print_n_valid(tensor, msg='n_valid: '):
    indices = tf.where(tf.math.not_equal(tensor, tf.constant(0, tensor.dtype)))
    n = tf.shape(indices)[0]
    # print(f'{msg} {indices.shape}')
    tf.Print(n, [n], msg)


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
    proxy_file_list = []
    proxy_r_file_list= []
    for l in lines:
        to_load = re.split(',|;',l.strip())
        left_file_list.append(to_load[0])
        right_file_list.append(to_load[1])
        if len(to_load)>2:
            gt_file_list.append(to_load[2])
        if len(to_load)>3:
            proxy_file_list.append(to_load[3])
        if len(to_load)>4:
            proxy_r_file_list.append(to_load[4])

    return left_file_list,right_file_list,gt_file_list,proxy_file_list, proxy_r_file_list

def read_image_from_disc(image_path,shape=None,dtype=tf.uint8):
    """
    Create a queue to hoold the paths of files to be loaded, then create meta op to read and decode image
    Args:
        image_path: metaop with path of the image to be loaded
        shape: optional shape for the image
    Returns:
        meta_op with image_data
    """         
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
        shuffle=False,
        exclusive=True):
    
        if not os.path.exists(path_file):
            raise Exception('File not found during dataset construction')
    
        self._path_file = path_file
        self._batch_size=batch_size
        self._crop_shape=crop_shape
        self._num_epochs=num_epochs
        self._augment=augment
        self._shuffle=shuffle
        self._is_training = is_training
        self._exclusive=exclusive
        self._build_input_pipeline_with_proxies()
    
    def _load_image_with_proxies(self, files):
        left_file_name = files[0]
        right_file_name = files[1]
        gt_file_name = files[2]
        proxy_file_name = files[3]
        proxy_r_file_name = files[4]


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
        proxy_image = read_image_from_disc(proxy_file_name,shape=[None,None,1], dtype=px_read_type)
        proxy_image = tf.cast(proxy_image,tf.float32)
        proxy_r_image = read_image_from_disc(proxy_r_file_name,shape=[None,None,1], dtype=px_read_type)
        proxy_r_image = tf.cast(proxy_r_image,tf.float32)

        if self._double_prec_px:
                proxy_image = proxy_image/256.0
                proxy_r_image = proxy_r_image / 256.0

        real_width = tf.shape(left_image)[1]

        proxy_image = proxy_image[:,:tf.shape(left_image)[1],:]
        # shape = proxy_image.shape
        proxy_r_image = proxy_r_image[:,:tf.shape(left_image)[1],:]



        if self._is_training:
            left_image,right_image,gt_image = preprocessing.random_crop(self._crop_shape, [left_image,right_image,gt_image])
        else:
            (left_image,right_image,gt_image,proxy_image,proxy_r_image) = [tf.image.resize_image_with_crop_or_pad(x,self._crop_shape[0],self._crop_shape[1]) for x in [left_image,right_image,gt_image,proxy_image,proxy_r_image]]

        #TODO: pgt sampling
        sparse_proxy, smask_l = self.disp_sample_by_percent(proxy_image)
        sparse_proxy_r, smask_r = self.disp_sample_by_percent(proxy_r_image)

        print_n_valid(proxy_image, 'before masking proxy: ')
        if self._exclusive:
            smask_l = tf.cast(smask_l,tf.float32)
            proxy_image = proxy_image * smask_l
            # proxy_image = tf.math.multiply(proxy_image, smask_l)
            # proxy_image = tf.boolean_mask(proxy_image, smask_l)
            print_n_valid(proxy_image, 'after masking proxy: ')

        if self._augment:
            left_image,right_image=preprocessing.augment(left_image,right_image)
        return [left_image,right_image,gt_image,proxy_image,sparse_proxy,sparse_proxy_r,real_width]

    def _build_input_pipeline_with_proxies(self):
        left_files, right_files, gt_files, proxy_files, proxy_r_files = read_list_file(self._path_file)
        self._couples = [[l, r, gt, px, rpx] for l, r, gt, px, rpx in zip(left_files, right_files, gt_files, proxy_files, proxy_r_files)]
        #flags 
        self._usePfm = gt_files[0].endswith('pfm') or gt_files[0].endswith('PFM')
        if not self._usePfm:
            gg = cv2.imread(gt_files[0],-1)
            self._double_prec_gt = (gg.dtype == np.uint16)

        gg = cv2.imread(proxy_files[0],-1)
        self._double_prec_px = (gg.dtype == np.uint16)

        #create dataset
        dataset = tf.data.Dataset.from_tensor_slices(self._couples).repeat(self._num_epochs)
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
        self._px_batch = images[3]
        self._s_px_batch = images[4]
        self._s_px_r_batch = images[5]
        self._real_width = images[6]

    # TODO: implement function
    def from_dense(self, tensor):
        # tensor = tf.convert_to_tensor(tensor)
        indices = tf.where(tf.math.not_equal(tensor, tf.constant(0,tensor.dtype)))
        values = tf.gather_nd(tensor, indices)
        shape = tf.shape(tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)

    def _random_choice(self, inputs, n_samples):
        """
        With replacement.
        Params:
          inputs (Tensor): Shape [n_states, n_features]
          n_samples (int): The number of random samples to take.
        Returns:
          sampled_inputs (Tensor): Shape [n_samples, n_features]
        """
        # (1, n_states) since multinomial requires 2D logits.
        # uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

        # ind = tf.multinomial(uniform_log_prob, n_samples)
        # ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

        ind = tf.random.shuffle(tf.range(tf.shape(inputs)[0]), 42)[:n_samples]
        idx = tf.nn.top_k(tf.cast(-ind,tf.int32), n_samples) # minus for ascending order
        idx = -idx.values

        return tf.gather(inputs, idx)
    def disp_sample_by_percent(self, disparity, sample_percent=0.3):
        h, w, _ = disparity.shape
        sparse = self.from_dense(disparity)
        # print(sparse)

        #TODO: sample pixels (later)
        # - randomly sample disparities and return a pgt as sparse matrix form with a mask(where invalid pixels are masked)
        # sample_indices = random choice (valid_indices, n_pixels)
        # indices = tf.cast(valid_indices, tf.float32)
        # disp_value = disparity[indices[:,0], indices[:,1],:]
        # sparse_pgt_uvz = tf.concat(indices, disp_value)

        valid_indices = sparse.indices
        # print_n_valid(valid_indices, 'valid_indices:')
        n = tf.cast(tf.shape(valid_indices)[0], tf.float32)
        n_pixels = tf.cast(tf.constant(sample_percent) * n,tf.int32)
        # print(f'n: {n} n_pixels: {n_pixels}')
        tf.Print(n, [n], "n: ")
        tf.Print(n_pixels, [n_pixels], "n_pixels: ")

        sample_indices = self._random_choice(valid_indices, n_pixels)

        # sample_indices = tf.multinomial(valid_indices,n_pixels,42)
        # sample_indices = tf.sort(sample_indices, 0)
        # tf.Print(sample_indices, [sample_indices[0],sample_indices[1],sample_indices[2], sample_indices[3]], "sample_indies: ")

        # sample_indices = valid_indices # delete later
        sample_values = tf.gather_nd(disparity, sample_indices)
        smask = tf.sparse_to_dense(sparse_indices=sample_indices,output_shape=disparity.shape, sparse_values=0, default_value=1) # mask for gt disparity
        # tf.Print(smask,[smask],"smask:")
        # print_n_valid(sample_indices, 'sample_indices:')
        # print_n_valid(sample_values, 'sample_values:')


        sample_indices = tf.cast(sample_indices, tf.float32) # TODO: why dimension 3????
        # valid_indices = sparse.indices # TODO: why dimension 3????

        # disp_value = sparse.values
        # valid_indices = tf.reshape(sparse.indices, [-1,2])
        sample_values = tf.reshape(sample_values,[-1,1])
        # print(valid_indices,disp_value)
        sparse_uvz = tf.concat([sample_indices,sample_values],-1)
        # print(valid_indices)
        # tf.Print(valid_indices, [valid_indices], "sparse tensor's valid indices: ")
        # tf.Print(disp_value, [disp_value], "sparse tensor's valid values: ")
        # tf.Print(sparse_uvz, [sparse_uvz], "sparse tensor's valid values: ")



        return sparse_uvz, smask


    ################# PUBLIC METHOD #######################

    def __len__(self):
        return len(self._couples)
    
    def get_max_steps(self):
        return (len(self)*self._num_epochs)//self._batch_size

    def get_batch(self):
        return self._left_batch,self._right_batch,self._gt_batch,self._px_batch,self._s_px_batch, self._s_px_r_batch,self._real_width
    
    def get_couples(self):
        return self._couples