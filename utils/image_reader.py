# A DUPLICATE OF https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/image_reader.py
import os
import numpy as np
import tensorflow as tf

IMG_MEAN = {
        'r':   123.68, 
        'g':   116.779, 
        'b':   103.939}   # RGB, refer to https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow/issues/30

def image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    
    return img, label

def image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[-3]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[-2]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, image_channel=3, label_channel=1, ignore_label=255):
    label = tf.cast(label, dtype=tf.float32)
    if ignore_label:
        label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=-1, values=[image, label])
    image_shape = tf.shape(image)[-3:]
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,int(image_channel+label_channel)])
    img_crop = combined_crop[..., :last_image_dim]
    label_crop = combined_crop[..., last_image_dim:]
    if ignore_label:
        label_crop = label_crop + ignore_label
        label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, image_channel))
    label_crop.set_shape((crop_h,crop_w, label_channel))
    return img_crop, label_crop

def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")

        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)

        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)

        if not tf.gfile.Exists(mask):
            raise ValueError('Failed to find file: ' + mask)

        images.append(image)
        masks.append(mask)

    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, img_mean, image_channel=3, img_channel_format='BGR', label_channel=1, ignore_label=255): # optional pre-processing arguments
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_jpeg(img_contents, channels=image_channel)
    img_r, img_g, img_b = tf.split(axis=-1, num_or_size_splits=image_channel, value=img)
    if img_channel_format.lower() == 'bgr':
        img = tf.cast(tf.concat(axis=-1, values=[img_b, img_g, img_r]), dtype=tf.float32)
    elif img_channel_format.lower() == 'brg':
        img = tf.cast(tf.concat(axis=-1, values=[img_b, img_r, img_g]), dtype=tf.float32)
    elif img_channel_format.lower() == 'gbr':
        img = tf.cast(tf.concat(axis=-1, values=[img_g, img_b, img_r]), dtype=tf.float32)
    elif img_channel_format.lower() == 'grb':
        img = tf.cast(tf.concat(axis=-1, values=[img_g, img_r, img_b]), dtype=tf.float32)
    elif img_channel_format.lower() == 'rgb':
        img = tf.cast(tf.concat(axis=-1, values=[img_r, img_g, img_b]), dtype=tf.float32)
    elif img_channel_format.lower() == 'rbg':
        img = tf.cast(tf.concat(axis=-1, values=[img_r, img_b, img_g]), dtype=tf.float32)
    else:
        raise NameError("No support %s format." % img_channel_format)
    # Extract mean.
    if img_mean is None:
        img_mean = np.array([IMG_MEAN[k] for k in img_channel_format.lower()], dtype=np.float32)
    img -= img_mean

    if label_channel == 1:
        label = tf.image.decode_png(label_contents, channels=label_channel)
    else:
        label = tf.image.decode_jpeg(label_contents, channels=label_channel)

    if input_size is not None:
        h, w = input_size

        if random_scale:
            img, label = image_scaling(img, label)

        if random_mirror:
            img, label = image_mirroring(img, label)
            
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, image_channel=image_channel, label_channel=label_channel, ignore_label=ignore_label)

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size,
                  random_scale, random_mirror, ignore_label, img_mean, coord, img_channel_format='BGR', rgb_label=False):

        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        if rgb_label:
            self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, img_mean, img_channel_format=img_channel_format, label_channel=3, ignore_label=None)
        else:
            self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, img_mean, img_channel_format=img_channel_format, label_channel=1, ignore_label=ignore_label)

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch
