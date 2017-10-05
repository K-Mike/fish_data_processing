import numpy as np
import pandas as pd
import shutil
import imageio
import os
from tqdm import tqdm
from PIL import Image as pil_image
from keras.preprocessing import image

from .utils import test_dir, train_dir, get_ruler_Crop_Area, crop_image_2


def video2lstm_images_train(df, dir_out, model_shape_in, model, preprocess_input, step=1, fr_width=3):

    x_tester = np.zeros((1, *model_shape_in))
    model_shape_out = model.predict(x_tester).shape

    filename = os.path.join(dir_out, 'X_train')
    # 1 good and 2 bad frames
    shape_out = (df.shape[0] * 3, fr_width) + model_shape_out[1:]
    X_train = np.memmap(filename, dtype='float32', mode='w+', shape=shape_out)
    Y_train = np.zeros((shape_out[0]))

    pbar = tqdm(total=df['hash'].unique().shape[0])
    crop_area_cache = {}
    counter = 0
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

        # get crop area
        video_path = os.path.join(train_dir, row.video_id + '.mp4')
        vid = imageio.get_reader(video_path, 'ffmpeg')
        if row.video_id not in crop_area_cache.keys():
            crop_area_cache[row.video_id] = get_ruler_Crop_Area(vid)

        crop_area = crop_area_cache[row.video_id]

        # save data
        lstm_frame = get_lstm_frame(vid, row.frame, step=step, fr_width=fr_width)
        X_train[counter] = model.predict(prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in))
        Y_train[counter] = 1.0
        counter += 1

        for i_fr in i_frame_bad:
            lstm_frame = get_lstm_frame(vid, i_fr, step=step, fr_width=fr_width)
            X_train[counter] = model.predict(prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in))
            Y_train[counter] = 0.0
            counter += 1

        pbar.update(1)

    np.save(os.path.join(dir_out, 'Y_train'), Y_train)


def video2lstm_images_test(dir_out, model_shape_in, model, preprocess_input, step=1, fr_width=3):

    x_tester = np.zeros((1, *model_shape_in))
    model_shape_out = model.predict(x_tester)

    filename = os.path.join(dir_out, 'X_train')
    # 1 good and 2 bad frames
    shape_out = (df.shape[0] * 3, *model_shape_out)
    X_train = np.memmap(filename, dtype='float32', mode='w+', shape=shape_out)
    Y_train = np.zeros((shape_out[0]))

    pbar = tqdm(total=df['hash'].unique().shape[0])
    crop_area_cache = {}
    counter = 0
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

        # get crop area
        video_path = os.path.join(train_dir, row.video_id + '.mp4')
        vid = imageio.get_reader(video_path, 'ffmpeg')
        if row.video_id not in crop_area_cache.keys():
            crop_area_cache[row.video_id] = get_ruler_Crop_Area(vid)

        crop_area = crop_area_cache[row.video_id]

        # save data
        lstm_frame = get_lstm_frame(vid, row.frame, step=step, fr_width=fr_width)
        X_train[counter] = prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in)
        Y_train[counter] = 1.0
        counter += 1

        for i_fr in i_frame_bad:
            lstm_frame = get_lstm_frame(vid, i_fr, step=step, fr_width=fr_width)
            X_train[counter] = prepare_lstm_frame(lstm_frame, crop_area, preprocess_input, model_shape_in)
            Y_train[counter] = 0.0
            counter += 1

        pbar.update(1)

    np.save(os.path.join(dir_out, 'Y_train'), Y_train)

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
        lstm_frame_crop[i] = preprocess_input(crop_img)

    return lstm_frame_crop



def get_lstm_frame(vid, i_frame, step=1, fr_width=3):
    vid_length = vid.get_length()
    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)

    frame_0 = vid.get_data(0)
    frame_arr = np.zeros((fr_width, *frame_0.shape))
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
