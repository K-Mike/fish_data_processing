import numpy as np
import pandas as pd
import shutil
import imageio
import os
import pickle
import glob
from tqdm import tqdm
from PIL import Image as pil_image
from keras.preprocessing import image
from skimage import io
from skimage import img_as_ubyte
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


from .utils import test_dir, train_dir, get_ruler_Crop_Area, crop_image_2



img_shape_dict = {'VGG16': (224, 224),
                  'Xception': (299, 299),
                  'VGG19': (224, 224),
                  'ResNet50': (224, 224),
                  'InceptionV3': (299, 299),
                  # 'InceptionResNetV2': (299, 299),
                  'MobileNet': (224, 224)
                  }

preprocess_input = {'VGG16': vgg16.preprocess_input,
                  'Xception': xception.preprocess_input,
                  'VGG19': vgg19.preprocess_input,
                  'ResNet50': resnet50.preprocess_input,
                  'InceptionV3': inception_v3.preprocess_input
                  }

def_model_name = 'hash_fish_detection_bottelneck_lstm'
bottelneck_dir = 'bottleneck_lstm'


class Bottleneck_LSTM():

    def __init__(self, base_model_name, model_name=None, pooling='avg', verbose=0):

        if base_model_name not in img_shape_dict.keys():
            print('Unknown model')
            raise Exception

        self.pooling = pooling
        self.base_model_name = base_model_name
        # define input size
        self.img_shape = img_shape_dict[base_model_name]

        if model_name is None:
            self.model_name = def_model_name + '_' + base_model_name
        else:
            self.model_name = model_name

        self.preprocess_input = preprocess_input[base_model_name]

        # find base_model output shape
        self.load_base_model()
        x_tester = np.zeros((1, *self.img_shape, 3))
        self.model_shape_out = self.base_model.predict(x_tester).shape
        del self.base_model

        # create model
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True, input_shape=(None, *self.model_shape_out[1:])))
        self.model.add(LSTM(32))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if verbose > 0:
            print(self.model.summary())

    def load_model(self, path=None):

        if path is None:
            def_path = os.path.join(bottelneck_dir, self.model_name + '.h5')
            self.model = load_model(def_path)
        else:
            self.model = load_model(path)

    def fit(self, X_train, Y_train, epochs, batch_size):
        self.model.fit(X_train, Y_train, epochs=epochs,
                  validation_split=0.2,
                  batch_size=batch_size, verbose=1, shuffle=True)

    def load_base_model(self):
        # create the base pre-trained model
        if self.base_model_name == 'VGG16':
            self.base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=self.pooling)
        elif self.base_model_name == 'Xception':
            self.base_model = xception.Xception(weights='imagenet', include_top=False, pooling=self.pooling)
        elif self.base_model_name == 'VGG19':
            self.base_model = vgg19.VGG19(weights='imagenet', include_top=False, pooling=self.pooling)
        elif self.base_model_name == 'ResNet50':
            self.base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling=self.pooling)
        elif self.base_model_name == 'InceptionV3':
            self.base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=self.pooling)
        elif self.base_model_name == 'MobileNet':
            self.base_model = mobilenet.MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False,
                                             pooling=self.pooling)

    def creat_train_dataset(self, train_frames, fr_width = 5, fr_step = 2):
        shape_out = (train_frames.shape[0], fr_width) + self.model_shape_out[1:]

        path = 'crop_areas_train.pkl'
        with open(path, 'rb') as f:
            crop_areas = pickle.load(f)

        X_train = np.zeros(shape_out, dtype='float32')
        Y_train = np.array(train_frames.y)



    def video_to_lstmframes(self, video_path, img_dir, path_out, fr_width=5, fr_step=2):
        vid = imageio.get_reader(video_path, 'ffmpeg')
        n_frames = vid.get_length()
        video_id = os.path.basename(video_path)[:-4]

        shape_out = (n_frames, fr_width) + self.model_shape_out[1:]
        # X_test_features = np.memmap(path_out, dtype='float32', mode='w+', shape=shape_out
        X_test_features = np.zeros(shape_out, dtype='float32')

        for i_frame in range(n_frames):
            lstm_frame = get_lstm_frame_dir(img_dir, video_id, n_frames, i_frame,
                                            step=fr_step, fr_width=fr_width)

            lstm_frame = prepare_lstm_frame_dir(lstm_frame, self.preprocess_input, self.img_shape + (3, ))
            features = self.base_model.predict(lstm_frame)
            X_test_features[i_frame] = features

        np.save(path_out, X_test_features)

    def create_dataset_from_dir(self, img_dir, vid_dir, dir_out, fr_width=5, fr_step=2):
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        os.makedirs(dir_out)

        self.load_base_model()

        vid_paths = glob.glob(vid_dir + '/*.mp4')

        pbar = tqdm(total=len(vid_paths))
        for video_path in vid_paths:
            video_id = os.path.basename(video_path)[:-4]
            path_out = os.path.join(dir_out, video_id)
            self.video_to_lstmframes(video_path, img_dir, path_out, fr_width=fr_width, fr_step=fr_step)

            pbar.update(1)

        del self.base_model

    def predict(self, X_test):
        return self.model.predict(X_test)



def video2lstm_images_train(hash_train_frames, dir_out, model_shape_in, model, preprocess_input, step=1, fr_width=3):
    x_tester = np.zeros((1, *model_shape_in))
    model_shape_out = model.predict(x_tester).shape

    shape_out = (hash_train_frames.unique().shape[0], fr_width) + model_shape_out[1:]
    filename = os.path.join(dir_out, 'X_train.dat')

    X_train = np.memmap(filename, dtype='float32', mode='w+', shape=shape_out)
    Y_train = np.zeros((shape_out[0]))

    pbar = tqdm(total=hash_train_frames.shape[0])
    crop_area_cache = {}
    for i, row in hash_train_frames.iterrows():



        pbar.update(1)

# def video2lstm_images_train(df, dir_out, model_shape_in, model, preprocess_input, step=1, fr_width=3):
#     # create table with video_ids and n_frames for cropping
#
#     x_tester = np.zeros((1, *model_shape_in))
#     model_shape_out = model.predict(x_tester).shape
#
#     filename = os.path.join(dir_out, 'X_train')
#     # 1 good and 2 bad frames
#     shape_out = (df['hash'].unique().shape[0] * 3, fr_width) + model_shape_out[1:]
#     X_train = np.memmap(filename, dtype='float32', mode='w+', shape=shape_out)
#     Y_train = np.zeros((shape_out[0]))
#
#     pbar = tqdm(total=df['hash'].unique().shape[0])
#     crop_area_cache = {}
#     counter = 0
#     for img_hash in df['hash'].unique():
#
#         fr_max = df[df['hash'] == img_hash].frame.max()
#         row = df[(df['hash'] == img_hash) & (df['frame'] == fr_max)].iloc[0]
#
#         fish_frames = np.array(df[df['video_id'] == row.video_id].frame)
#         fish_frames.sort(axis=0)
#         ind = np.where(fish_frames == row.frame)[0][0]
#
#         i_frame_bad = []
#         if ind > 0:
#             n_i = int((fish_frames[ind] + fish_frames[ind - 1]) / 2)
#             i_frame_bad.append(n_i)
#
#         if fish_frames.shape[0] > ind + 1:
#             n_i = int((fish_frames[ind + 1] + fish_frames[ind]) / 2)
#             i_frame_bad.append(n_i)
#
#         if len(i_frame_bad) != 2:
#             print(' len(i_frame_bad) != 2', len(i_frame_bad))
#     #     # get crop area
#     #     video_path = os.path.join(train_dir, row.video_id + '.mp4')
#     #     vid = imageio.get_reader(video_path, 'ffmpeg')
#     #     if row.video_id not in crop_area_cache.keys():
#     #         crop_area_cache[row.video_id] = get_ruler_Crop_Area(vid)
#     #
#     #     crop_area = crop_area_cache[row.video_id]
#     #
#     #     # save data
#     #     lstm_frame = get_lstm_frame(vid, row.frame, step=step, fr_width=fr_width)
#     #     X_train[counter] = model.predict(prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in))
#     #     Y_train[counter] = 1.0
#     #     counter += 1
#     #
#     #     for i_fr in i_frame_bad:
#     #         lstm_frame = get_lstm_frame(vid, i_fr, step=step, fr_width=fr_width)
#     #         X_train[counter] = model.predict(prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in))
#     #         Y_train[counter] = 0.0
#     #         counter += 1
#     #
#     #     pbar.update(1)
#     #
#     # np.save(os.path.join(dir_out, 'Y_train'), Y_train)



def prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, img_shape):

    lstm_frame_crop = np.zeros((lstm_frame.shape[0], ) + img_shape)
    for i in range(lstm_frame.shape[0]):
        img = lstm_frame[i]
        crop_img = crop_image_2(img, crop_area)

        crop_img = pil_image.fromarray(np.uint8(crop_img))
        hw_tuple = (img_shape[1], img_shape[0])
        if crop_img.size != hw_tuple:
            crop_img = crop_img.resize(hw_tuple)

        crop_img = image.img_to_array(crop_img)
        # lstm_frame_crop[i] = preprocess_input(crop_img)
        lstm_frame_crop[i] = crop_img

    return lstm_frame_crop



def get_lstm_frame(vid, i_frame, step=1, fr_width=3):
    vid_length = vid.get_length()
    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)

    frame_0 = vid.get_data(0)
    frame_arr = np.zeros((fr_width, *frame_0.shape), dtype='uint8')
    for i, rel_ind in enumerate(rel_ind_range):
        i_fr = i_frame + rel_ind
        # Achtung sometimes it's not true
        if i_fr < 0 or i_fr >= vid_length:
            i_fr = i_frame + (-1) * rel_ind

        frame_arr[i] = vid.get_data(i_fr)

    return frame_arr


def get_best_video_id_hash(df, img_hash, step=1, fr_width=3):
    if fr_width % 2 == 0:
        print('Achtung, must be fr_width % 2 != 0')
        return

    df_hash_cur = df[df['hash'] == img_hash].copy()

    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)
    best_vid_score = 0
    best_video_id = None
    best_i_frame = 0
    for i_row, row in df_hash_cur.iterrows():

        video_path = os.path.join(train_dir, row.video_id + '.mp4')
        vid = imageio.get_reader(video_path, 'ffmpeg')
        vid_length = vid.get_length()

        fr_indexes = [rel_ind + row.frame for rel_ind in rel_ind_range if
                      rel_ind + row.frame > 0 and rel_ind + row.frame < vid_length]

        if best_vid_score < len(fr_indexes) or best_video_id is None:
            best_vid_score = len(fr_indexes)
            best_video_id = row.video_id
            best_i_frame = row.frame

        if best_vid_score == fr_width:
            break

    return best_video_id, best_i_frame


def get_bad_frames(df, vid, video_id, n_bad_frames=2):
    i_frames = list(range(vid.get_length()))
    for i_fr in df[df.video_id == video_id].frame:
        i_frames.remove(i_fr)

    frame_arr = []
    for _ in range(1000):
        i_fr = np.random.choice(i_frames, 1)[0]
        frame_arr.append(i_fr)
        i_frames.remove(i_fr)

        if len(frame_arr) == n_bad_frames:
            break

    return frame_arr


def save_lstm_frame(lstm_frame, path, i_cl):
    n_sub_fr = lstm_frame.shape[0]

    frame_list_crop = []
    for i in range(n_sub_fr):
        frame_crop = crop_ruler_area_short(lstm_frame[i], i_cl)
        frame_list_crop.append(frame_crop)

    frame_stack_crop = np.zeros((n_sub_fr, *frame_list_crop[0].shape))

    for i, fr in enumerate(frame_list_crop):
        frame_stack_crop[i] = fr

    np.save(path, frame_stack_crop.astype('float32'))


def get_lstm_frame_dir(dir_name, video_id, vid_length, i_frame, step=1, fr_width=3):

    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)

    fname = '_'.join([video_id, str(i_frame).zfill(5)]) + '.png'
    path = os.path.join(dir_name, fname)
    frame_0 = io.imread(path)[:, :, 0:3]
    frame_arr = np.zeros((fr_width, *frame_0.shape))
    for i, rel_ind in enumerate(rel_ind_range):
        i_fr = i_frame + rel_ind
        # Achtung sometimes it's not true
        if i_fr < 0 or i_fr >= vid_length:
            i_fr = i_frame + (-1) * rel_ind

        fname = '_'.join([video_id, str(i_fr).zfill(5)]) + '.png'
        path = os.path.join(dir_name, fname)
        frame_arr[i] = io.imread(path)[:, :, 0:3]

    return frame_arr


def prepare_lstm_frame_dir(lstm_frame, preprocess_input, img_shape):

    lstm_frame_crop = np.zeros((lstm_frame.shape[0], ) + img_shape)
    for i in range(lstm_frame.shape[0]):
        img = lstm_frame[i]

        crop_img = pil_image.fromarray(np.uint8(img))
        hw_tuple = (img_shape[1], img_shape[0])
        if crop_img.size != hw_tuple:
            crop_img = crop_img.resize(hw_tuple)

        crop_img = image.img_to_array(crop_img)
        lstm_frame_crop[i] = preprocess_input(crop_img)

    return lstm_frame_crop