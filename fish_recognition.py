import os
import imageio
from tqdm import tqdm
import shutil
import pandas as pd
import glob

from .utils import train_dir, test_dir


translate_dict = {'species_fourspot': 0,
                  'species_grey sole': 1,
                  'species_other': 2,
                  'species_plaice': 3,
                  'species_summer': 4,
                  'species_windowpane': 5,
                  'species_winter': 6}


def resort_images(data_train, dir_out, dirs_in, split=0.8):
    files = []
    for dir_in in dirs_in:
        files.extend(glob.glob(dir_in + '/*.png'))

    columns = list(data_train.columns)
    columns.append('path')
    df = pd.DataFrame(columns=columns)

    for path in files:
        video_id, frame = os.path.basename(path)[:-4].split('_')
        frame = int(frame)
        row = data_train[(data_train['video_id'] == video_id) & (data_train['frame'] == frame)]
        row = row.assign(path=[path])
        df = df.append(row, ignore_index=True)

    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    # train
    train_dir_out = os.path.join(dir_out, 'train')
    os.makedirs(train_dir_out)

    train_dir_dic = {}
    for species_name in translate_dict.keys():
        train_fish_dir = os.path.join(train_dir_out, species_name)
        os.makedirs(train_fish_dir)
        train_dir_dic[species_name] = train_fish_dir

    # validation
    val_dir = os.path.join(dir_out, 'val')
    os.makedirs(val_dir)

    val_dir_dic = {}
    for species_name in translate_dict.keys():
        val_fish_dir = os.path.join(val_dir, species_name)
        os.makedirs(val_fish_dir)
        val_dir_dic[species_name] = val_fish_dir

    # split
    def copy_fiels(df, dir_out):
        for i, row in df.iterrows():
            fname = '_'.join([row.video_id, str(row.frame)]) + '.png'
            dst = os.path.join(dir_out, fname)
            shutil.copyfile(row.path, dst)

    pbar = tqdm(total=len(translate_dict.keys()))
    for sp in translate_dict.keys():
        df_sub = df[df[sp] == 1]
        n = df_sub.shape[0]
        n_train = int(n * split)
        train_df = df_sub[:n_train]
        copy_fiels(train_df, train_dir_dic[sp])
        validation_df = df_sub[n_train:]
        copy_fiels(validation_df, val_dir_dic[sp])

        pbar.update(1)
