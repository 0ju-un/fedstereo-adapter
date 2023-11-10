import tensorflow as tf

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
    lines = [x for x in lines if not (x.strip() == '' or x.strip()[0] == '#')]
    left_file_list = []
    right_file_list = []
    gt_file_list = []
    conf_file_list = []
    for l in lines:
        to_load = re.split(',|;',l.strip())
        left_file_list.append(to_load[0])
        right_file_list.append(to_load[1])
        if len(to_load)>2:
            gt_file_list.append(to_load[2])
        if len(to_load)>3:
            conf_file_list.append(to_load[3])
    return left_file_list,right_file_list,gt_file_list,conf_file_list

def read_image_from_disc(image_path,shape=None,dtype=tf.uint8):
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

class Dataloader(object):

    def __init__(self, dataset, left_dir, right_dir, disp_dir, r_disp=False):

        self.dataset = dataset

        self.left = None
        self.right = None
        self.disp  = None

        input_queue = tf.train.string_input_producer([self.dataset], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        files = tf.string_split([line], ',').values
        split_line = tf.string_split([line], '/').values

        left_img = tf.stack([tf.cast(self.read_image(files[0], [None, None, 3]), tf.float32)], 0)
        right_img = tf.stack([tf.cast(self.read_image(files[1], [None, None, 3]), tf.float32)], 0)

        # for right disparity
        if r_disp:
            self.left = tf.image.flip_left_right(right_img)
            self.right = tf.image.flip_left_right(left_img)
            self.disp_path = files[4]
            self.disp = tf.image.flip_left_right(tf.stack([tf.cast(self.read_image(self.disp_path, [None, None, 1], dtype=tf.uint16), tf.float32)], 0) / 256.)
        else:
            self.left = left_img
            self.right = right_img
            self.disp_path = files[3]
            self.disp = tf.stack([tf.cast(self.read_image(self.disp_path, [None, None, 1], dtype=tf.uint16), tf.float32)], 0) / 256.
        self.filename = split_line[-1]


    def read_image(self, image_path, shape=None, dtype=tf.uint8, norm=False):
        image_raw = tf.read_file(image_path)
        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)
        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)

        return image
