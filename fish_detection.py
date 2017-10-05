import os
import shutil
from tqdm import tqdm
import numpy as np
import imageio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing import image
import glob
from .utils import crop_ruler_area_short, get_ruler_Crop_Area, crop_image_2

#
# test_dir = 'test_videos'
# train_dir = 'train_videos'

test_dir = '/mnt/nvme/jupyter/ddata_fish/test_videos'
train_dir = '/mnt/nvme/jupyter/ddata_fish/train_videos'


def video2images_train(df, dir_out, split=0.8):
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    # train
    train_dir_out = os.path.join(dir_out, 'train')
    os.makedirs(train_dir_out)

    train_fish_dir = os.path.join(train_dir_out, 'fish')
    os.makedirs(train_fish_dir)

    train_nonfish_dir = os.path.join(train_dir_out, 'nonfish')
    os.makedirs(train_nonfish_dir)

    # validation
    val_dir = os.path.join(dir_out, 'val')
    os.makedirs(val_dir)

    val_fish_dir = os.path.join(val_dir, 'fish')
    os.makedirs(val_fish_dir)

    val_nonfish_dir = os.path.join(val_dir, 'nonfish')
    os.makedirs(val_nonfish_dir)

    pbar = tqdm(total=df['hash'].unique().shape[0])
    crop_area_cache = {}
    for img_hash in df['hash'].unique():

        fr_max = df[df['hash'] == img_hash].frame.max()
        row = df[(df['hash'] == img_hash) & (df['frame'] == fr_max)].iloc[0]

        fish_frames = np.array(df[df['video_id'] == row.video_id].frame)
        fish_frames.sort(axis=0)
        ind = np.where(fish_frames == row.frame)[0][0]

        i_frame_bad = []
        if ind > 0:
            n_i = int((fish_frames[ind] + fish_frames[ind - 1]) / 2)
            i_frame_bad.append(n_i)

        if fish_frames.shape[0] > ind + 1:
            n_i = int((fish_frames[ind + 1] + fish_frames[ind]) / 2)
            i_frame_bad.append(n_i)

        video_path = os.path.join(train_dir, row.video_id + '.mp4')
        vid = imageio.get_reader(video_path, 'ffmpeg')
        if row.video_id not in crop_area_cache.keys():
            crop_area_cache[row.video_id] = get_ruler_Crop_Area(vid)

        crop_area = crop_area_cache[row.video_id]

        choise = np.random.choice(['validation', 'train'], p=[1 - split, split])
        if choise == 'train':
            frame = vid.get_data(row.frame)
            crop_img = crop_image_2(frame, crop_area)

            fname = '_'.join([row.video_id, str(row.frame).zfill(4)]) + '.png'
            path = os.path.join(train_fish_dir, fname)
            plt.imsave(path, crop_img)

            for i_fr in i_frame_bad:
                frame = vid.get_data(i_fr)
                crop_img = crop_image_2(frame, crop_area)

                fname = '_'.join([row.video_id, str(i_fr).zfill(4)]) + '.png'
                path = os.path.join(train_nonfish_dir, fname)
                plt.imsave(path, crop_img)

        else:
            frame = vid.get_data(row.frame)
            crop_img = crop_image_2(frame, crop_area)

            fname = '_'.join([row.video_id, str(row.frame).zfill(4)]) + '.png'
            path = os.path.join(val_fish_dir, fname)
            plt.imsave(path, crop_img)

            for i_fr in i_frame_bad:
                frame = vid.get_data(i_fr)
                crop_img = crop_image_2(frame, crop_area)

                fname = '_'.join([row.video_id, str(i_fr).zfill(4)]) + '.png'
                path = os.path.join(val_nonfish_dir, fname)
                plt.imsave(path, crop_img)

        pbar.update(1)


def video2images_test(dir_out):

    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    videos = glob.glob(test_dir + '/*.mp4')
    pbar = tqdm(total=len(videos))
    for video_path in videos:
        video_id = os.path.basename(video_path)[:-4]
        vid = imageio.get_reader(video_path, 'ffmpeg')
        crop_area = get_ruler_Crop_Area(vid)

        vid = imageio.get_reader(video_path, 'ffmpeg')
        for i_fr, frame in enumerate(vid):
            #         crop_img = crop_image(frame, crop_area)
            crop_img = crop_image_2(frame, crop_area)
            fname = '_'.join([video_id, str(i_fr).zfill(5)]) + '.png'
            path_out = os.path.join(dir_out, fname)

            plt.imsave(path_out, crop_img)

        pbar.update(1)


def create_simple_CNN(img_shape):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(*img_shape, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    # model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def create_like_vgg16_CNN(img_shape):
    input_shape = (*img_shape, 3)
    img_input = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), activation='relu', name='block1_conv1')(img_input)
    x = Conv2D(8, (3, 3), activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)

    x = Conv2D(8, (3, 3), activation='relu', name='block2_conv1')(x)
    x = Conv2D(8, (3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)

    x = Conv2D(16, (3, 3), activation='relu', name='block3_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax', name='fc2')(x)

    model = Model(img_input, x, name='vgg16')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def create_like_vgg16_CNN_batchnorm(img_shape):
    input_shape = (*img_shape, 3)
    img_input = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), activation=None, name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), activation=None, name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)

    x = Conv2D(8, (3, 3), activation=None, name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), activation=None, name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)

    x = Conv2D(16, (3, 3), activation=None, name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), activation=None, name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, activation=None, name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation=None, name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation=None, name='pred')(x)
    x = BatchNormalization()(x)
    # x = Activation('softmax')(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x, name='vgg16')

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def create_like_vgg16_CNN_batchnorm_bin(img_shape, fl_size=(3, 3)):
    input_shape = (*img_shape, 3)
    img_input = Input(shape=input_shape)

    x = Conv2D(16, fl_size, activation=None, name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, fl_size, activation=None, name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)

    x = Conv2D(32, fl_size, activation=None, name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, fl_size, activation=None, name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)

    # x = Conv2D(16, fl_size, activation=None, name='block3_conv1')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, fl_size, activation=None, name='block3_conv2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(32, activation=None, name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation=None, name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation=None, name='pred')(x)
    x = BatchNormalization()(x)
    # x = Activation('softmax')(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x, name='vgg16_bin')

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model