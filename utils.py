import numpy as np
import os
import numpy as np
import imageio
from collections import namedtuple
from skimage import transform as tf
import pandas as pd
import importlib.machinery
import imagehash
from PIL import Image
from collections import Counter

df_train_cl = pd.read_csv('/mnt/data/jupyter/ddata_fish/data_train_hash.csv', index_col=0)
image_utils = importlib.machinery.SourceFileLoader('image_utils','/mnt/data/jupyter/ddata_fish/image_utils/__init__.py').load_module()
# from  mnt/data/jupyter/ddata_fish/image_utils import find_ruler_contours, get_contour_groups, ContourGroup

test_dir = 'test_videos'
train_dir = 'train_videos'
img_shape = (1280, 720)

test_video_ids = ['0EmM5wsVVNqaKNaM',
                  '0Vn7LRp72VjFggGy',
                  '1uI5nLvyJxt9sHfh',
                  '0agWG0Rmk8SIeSsf',
                  '2MoPcsoPPO6v0VJ0',
                  'ZVg07UANwbxMkOpk',
                  '0RBt5mjWuPIpTlAq',
                  'T6jABNzRERDxIlpj',
                  '00WK7DR6FyPZ5u3A']



Crop_Area = namedtuple('Crop_Area', ['x', 'y', 'w', 'h', 'angle_rad'])


# crop_areas = {0: Crop_Area(x=509, y=106, w=225, h=547, angle_rad=0.014148761517249734),
#               1: Crop_Area(x=288, y=649, w=270, h=627, angle_rad=0.032292617822046302),
#               2: Crop_Area(x=142, y=1086, w=542, h=245, angle_rad=0.05041355562402261),
#               3: Crop_Area(x=264, y=556, w=215, h=524, angle_rad=0.91633271015635309)}

# class_video_id = {0: 'zqWCbHoUPI9lMVBK',
#                  1: 'zR6VkJBSdt1ZrAEx',
#                  2: 'n0DPcHASUSf0TGpj',
#                  3: 'p4JyjnPVsAlzZCD0'}

crop_areas = {0: Crop_Area(x=600, y=107, w=47, h=596, angle_rad=0.017926298119140965),
             1: Crop_Area(x=393, y=645, w=61, h=636, angle_rad=0.033578995413217176),
             2: Crop_Area(x=30, y=1179, w=534, h=52, angle_rad=0.033802656331878289),
             3: Crop_Area(x=346, y=557, w=46, h=528, angle_rad=0.91382810350511656)}


def get_mean_image(video_iterator):
    avg_img = None
    for image in video_iterator:
        if avg_img is None:
            avg_img = np.array(image, dtype=float)
        else:
            avg_img += np.array(image, dtype=float)

    avg_img /= len(video_iterator)

    return avg_img.astype(np.uint8)


def get_train_frames(video_id, i_frames):

    video_path = os.path.join(train_dir, video_id + '.mp4')
    reader = imageio.get_reader(video_path, 'ffmpeg')
    img_shape = reader.get_data(0).shape

    frames = np.zeros((len(i_frames), *img_shape))
    for i, ind in enumerate(i_frames):
        fr = reader.get_data(ind)
        frames[i] = fr

    return frames


def get_coord_ruler(df, video_id, i_fr, crop_shape):

    if i_fr < 0:
        x_1 = df[df.video_id == video_id].dropna().x1.mean()
        y_1 = df[df.video_id == video_id].dropna().y1.mean()
        x_2 = df[df.video_id == video_id].dropna().x2.mean()
        y_2 = df[df.video_id == video_id].dropna().y2.mean()
    else:
        x_1 = df[(df.video_id == video_id) & (df.frame == i_fr)].dropna().iloc[0].x1
        y_1 = df[(df.video_id == video_id) & (df.frame == i_fr)].dropna().iloc[0].y1
        x_2 = df[(df.video_id == video_id) & (df.frame == i_fr)].dropna().iloc[0].x2
        y_2 = df[(df.video_id == video_id) & (df.frame == i_fr)].dropna().iloc[0].y2

    x_m, y_m = (x_1 + x_2) / 2, (y_1 + y_2) / 2
    x1 = int(x_m - crop_shape[0] / 2)
    y1 = int(y_m - crop_shape[1] / 2)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    if x1 + crop_shape[0] >= img_shape[0]:
        x1 -= (x1 + crop_shape[0] - img_shape[0])

    if y1 + crop_shape[1] >= img_shape[1]:
        y1 -= (y1 + crop_shape[1] - img_shape[1])

    return Crop_Area(x=x1, y=y1, w=crop_shape[0], h=crop_shape[1])


def get_coord_non_ruler(img_shape, crop_area, n_areas=2):
    N = 1000

    areas = []
    for _ in range(N):
        x = np.random.randint(0, img_shape[0] - crop_area.w, size=1)[0]
        y = np.random.randint(0, img_shape[1] - crop_area.h, size=1)[0]

        cur_area = Crop_Area(x=x, y=y, w=crop_area.w, h=crop_area.h)

        if len(areas) != 0:
            inter = [area(a, cur_area) / (a.w * a.h) for a in areas]
            if max(inter) > 0.3:
                continue

        if area(crop_area, cur_area) / (crop_area.w * crop_area.h) < 0.05:
            areas.append(cur_area)

        if len(areas) == n_areas:
            break

    return areas


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    dy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)

    area = 0
    if (dx >= 0) and (dy >= 0):
        area = dx * dy

    return area


def correlation_coefficient(map_1, map_2):
    """
    Calculate correlation coefficient between two images.
    :param map_1: numpy array.
    :param map_2: numpy array.
    :return: float
    """
    product = np.mean((map_1 - map_1.mean()) * (map_2 - map_2.mean()))
    stds = map_1.std() * map_2.std()

    product = product / stds if stds != 0 else 0

    return product


def crop_ruler_area_short(frame, i_cl):
    img_rot = tf.rotate(frame.copy(), crop_areas[i_cl]['angle'] * 180 / np.pi)
    crop_area = crop_areas[i_cl]['crop_area']
    x, w, y, h = crop_area.x, crop_area.w, crop_area.y, crop_area.h

    if h > w:
        crop_img = img_rot[x:x + w, y:y + h]
    else:
        crop_img = img_rot[x:x + w, y:y + h]
        crop_img = np.rot90(crop_img)

    return crop_img


def get_ruler_Crop_Area(vid):
    try:
        # cnt_groups = image_utils.get_contour_groups(vid, show_result_image=False, line_angle_max_error=None)
        cnt_groups = image_utils.get_contour_groups_parallel(vid, show_result_image=False, line_angle_max_error=None)
    except Exception as e:
        frame_0 = vid.get_data(0)
        cl = get_class_sim(df_train_cl, frame_0)

        return crop_areas[cl]


    # transform coordinates of all contours
    x0, y0 = [el / 2 for el in vid.get_meta_data()['size'][::-1]]
    # angle_rad = np.arctan(cnt_groups[0].fitted_line[0])
    angle_rad = np.mean([cnt_groups[i].fitted_line[0] for i in range(len(cnt_groups))])
    angle_rad = np.arctan(angle_rad)
    # angle_rad = min((np.pi / 2) - angle_rad, angle_rad)
    # if np.abs(angle_rad) < np.pi / 18:
    #     angle_rad = 0.0
    # if np.abs(np.pi - np.abs(angle_rad)) < np.pi / 10:
    if angle_rad < 0:
        angle_rad = (np.pi / 2) - np.abs(angle_rad)

    if np.pi / 2 - angle_rad < 0.05:
        angle_rad = np.pi / 2 - angle_rad

    x_arr = []
    y_arr = []
    for contour_group in cnt_groups:
        for contor in contour_group.contours:
            x_arr.extend(contor[:, :, 1].flatten())
            y_arr.extend(contor[:, :, 0].flatten())

    x_arr_tr = [((x1 - x0) * np.cos(angle_rad)) - ((y1 - y0) * np.sin(angle_rad)) + x0 for x1, y1 in zip(x_arr, y_arr)]
    y_arr_tr = [((x1 - x0) * np.sin(angle_rad)) + ((y1 - y0) * np.cos(angle_rad)) + y0 for x1, y1 in zip(x_arr, y_arr)]

    # get coordinates of crop rectangle (x, y) min point (w, h) width and height for extending
    x, y = min(x_arr_tr), min(y_arr_tr)
    w, h = max(x_arr_tr) - min(x_arr_tr), max(y_arr_tr) - min(y_arr_tr)
    x, y, w, h = int(max(x, 0)), int(max(y, 0)), int(w), int(h)

    crop_area = Crop_Area(x, y, w, h, angle_rad)

    return crop_area


def get_ruler_Crop_Area_fast(cnt_groups, vid):
    if cnt_groups is None or len(cnt_groups) == 0:
        frame_0 = vid.get_data(0)
        cl = get_class_sim(df_train_cl, frame_0)

        return crop_areas[cl]

    # transform coordinates of all contours
    x0, y0 = [el / 2 for el in vid.get_meta_data()['size'][::-1]]
    # angle_rad = np.arctan(cnt_groups[0].fitted_line[0])
    angle_rad = np.mean([cnt_groups[i].fitted_line[0] for i in range(len(cnt_groups))])
    angle_rad = np.arctan(angle_rad)

    # angle_rad = min((np.pi / 2) - angle_rad, angle_rad)
    # if np.abs(angle_rad) < np.pi / 18:
    #     angle_rad = 0.0
    # if np.abs(np.pi - np.abs(angle_rad)) < np.pi / 10:
    if angle_rad < 0:
        angle_rad = (np.pi / 2) - np.abs(angle_rad)

    if np.pi / 2 - angle_rad < 0.05:
        angle_rad = np.pi / 2 - angle_rad

    x_arr = []
    y_arr = []
    for contour_group in cnt_groups:
        for contor in contour_group.contours:
            x_arr.extend(contor[:, :, 1].flatten())
            y_arr.extend(contor[:, :, 0].flatten())

    x_arr_tr = [((x1 - x0) * np.cos(angle_rad)) - ((y1 - y0) * np.sin(angle_rad)) + x0 for x1, y1 in zip(x_arr, y_arr)]
    y_arr_tr = [((x1 - x0) * np.sin(angle_rad)) + ((y1 - y0) * np.cos(angle_rad)) + y0 for x1, y1 in zip(x_arr, y_arr)]

    # get coordinates of crop rectangle (x, y) min point (w, h) width and height for extending
    x, y = min(x_arr_tr), min(y_arr_tr)
    w, h = max(x_arr_tr) - min(x_arr_tr), max(y_arr_tr) - min(y_arr_tr)
    x, y, w, h = int(max(x, 0)), int(max(y, 0)), int(w), int(h)

    crop_area = Crop_Area(x, y, w, h, angle_rad)

    return crop_area


def crop_image(img, crop_area, k=2):
    img_rot = tf.rotate(img.copy(), crop_area.angle_rad * 180 / np.pi)
    x, w, y, h = crop_area.x, crop_area.w, crop_area.y, crop_area.h

    if h > w:
        x -= k * w
        x = 0 if x < 0 else min(x, img.shape[0])
        w += (2 * k * w)
        crop_img = img_rot[x:x + w, y:y + h]
    else:
        y -= k * h
        y = 0 if y < 0 else min(y, img.shape[1])
        h += (2 * k * h)
        crop_img = img_rot[x:x + w, y:y + h]
        crop_img = np.rot90(crop_img)

    return crop_img


def crop_image_2(img, crop_area, k=2, kh=0.05):
    img_rot = tf.rotate(img.copy(), crop_area.angle_rad * 180 / np.pi, resize=True, mode='wrap')
    # img_rot = tf.rotate(img.copy(), crop_area.angle_rad * 180 / np.pi, resize=True, mode='symmetric')
    shape_dif = np.array(img_rot.shape[0:2]) - np.array(img.shape[0:2])
    x, w, y, h = crop_area.x + int(shape_dif[0] / 2), crop_area.w, crop_area.y + int(shape_dif[1] / 2), crop_area.h

    if h > w:
        x -= int(k * w)
        x = 0 if x < 0 else min(x, img.shape[0])
        w += int(2 * k * w)

        y -= int(kh * h)
        y = 0 if y < 0 else min(y, img.shape[1])
        h += int(2 * kh * h)
        crop_img = img_rot[x:x + w, y:y + h]
    else:
        y -= int(k * h)
        y = 0 if y < 0 else min(y, img.shape[1])
        h += int(2 * k * h)

        x -= int(kh * w)
        x = 0 if x < 0 else min(x, img.shape[0])
        w += int(2 * kh * w)
        crop_img = img_rot[x:x + w, y:y + h]
        crop_img = np.rot90(crop_img)

    return crop_img

def get_class_sim(df, frame):
    img_hash = imagehash.whash(Image.fromarray(frame), hash_size=32)
    X = df[['class', 'hash']].copy()

    data = {'class': -1, 'hash': str(img_hash)}
    X = X.append(data, ignore_index=True)
    X = X.sort_values(by=['hash', 'class']).reset_index(drop=True)
    ind = X[X['class'] == -1].index[0]

    c = Counter(X.iloc[ind - 3:ind]['class'])

    return c.most_common()[0][0]
