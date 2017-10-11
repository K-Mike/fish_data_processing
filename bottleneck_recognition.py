from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input, Dropout
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from PIL import Image as pil_image
from .utils import get_ruler_Crop_Area, Crop_Area, crop_image, get_class_sim, crop_areas
from .Histories import Histories
import os
import glob
from tqdm import tqdm

from .fish_recognition import translate_dict

bottelneck_dir = 'bottleneck'

img_shape_dict = {'VGG16': (224, 224),
                  'Xception': (299, 299),
                  'VGG19': (224, 224),
                  'ResNet50': (224, 224),
                  'InceptionV3': (299, 299),
                  # 'InceptionResNetV2': (299, 299),
                  'MobileNet': (224, 224)
                  }

def_model_name = 'hash_fish_recognition_bottelneck'


class Bottleneck_recognition():

    def __init__(self, base_model, model_name=None, pooling=None, verbose=0):
        """

        :param base_model: [vgg_16]
        :param model_name:
        :param verbose:
        """
        if base_model not in img_shape_dict.keys():
            print('Unknown model')
            raise Exception

        # define input size
        self.img_shape = img_shape_dict[base_model]

        if model_name is None:
            self.model_name = def_model_name + '_' + base_model
        else:
            self.model_name = model_name
        # this is the augmentation configuration we will use for training
        # and
        # this is the augmentation configuration we will use for testing:
        if base_model == 'VGG16':
            self.train_datagen, self.test_datagen = get_datagens(vgg16.preprocess_input)
            self.preprocess_input = vgg16.preprocess_input
        elif base_model == 'Xception':
            self.train_datagen, self.test_datagen = get_datagens(xception.preprocess_input)
            self.preprocess_input = xception.preprocess_input
        elif base_model == 'VGG19':
            self.train_datagen, self.test_datagen = get_datagens(vgg19.preprocess_input)
            self.preprocess_input = vgg19.preprocess_input
        elif base_model == 'ResNet50':
            self.train_datagen, self.test_datagen = get_datagens(resnet50.preprocess_input)
            self.preprocess_input = resnet50.preprocess_input
        elif base_model == 'InceptionV3':
            self.train_datagen, self.test_datagen = get_datagens(inception_v3.preprocess_input)
            self.preprocess_input = inception_v3.preprocess_input
        elif base_model == 'MobileNet':
            self.train_datagen, self.test_datagen = get_datagens(mobilenet.preprocess_input)
            self.preprocess_input = mobilenet.preprocess_input

        # create the base pre-trained model
        if base_model == 'VGG16':
            base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling=pooling)
        elif base_model == 'Xception':
            base_model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
        elif base_model == 'VGG19':
            base_model = vgg19.VGG19(weights='imagenet', include_top=False, pooling=pooling)
        elif base_model == 'ResNet50':
            base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling=pooling)
        elif base_model == 'InceptionV3':
            base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        elif base_model == 'MobileNet':
            base_model = mobilenet.MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling=pooling)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(translate_dict.keys()), activation='softmax')(x)

        # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(loss='categorical_crossentropy',
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

    def fit_generator(self, data_train_dir, data_valid_dir, batch_size, epochs, steps_per_epoch, validation_steps):
        # create data generators
        self.train_generator = self.train_datagen.flow_from_directory(
            data_train_dir,
            target_size=self.img_shape,
            batch_size=batch_size,
            class_mode='categorical')

        # this is a similar generator, for validation data
        self.validation_generator = self.test_datagen.flow_from_directory(
            data_valid_dir,
            target_size=self.img_shape,
            batch_size=batch_size,
            class_mode='categorical')

        # create callbacks
        def_path_log = os.path.join(bottelneck_dir, self.model_name + '.csv')
        log_history = Histories(def_path_log)
        def_path_model = os.path.join(bottelneck_dir, self.model_name + '.h5')
        checkpointer = ModelCheckpoint(filepath=def_path_model, verbose=1, save_best_only=True, monitor='val_acc')

        history = self.model.fit_generator(
                                self.train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                validation_data=self.validation_generator,
                                validation_steps=validation_steps,
                                callbacks=[checkpointer, log_history]
                            )

        return history

    def prepare_img(self, image_in):
        if image_in.max() > 1.0:
            img = pil_image.fromarray(np.uint8(image_in))
        else:
            img = pil_image.fromarray(np.uint8(image_in * 255))

        hw_tuple = (self.img_shape[1], self.img_shape[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)

        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = vgg16.preprocess_input(x)

        return x

    def load_img(self, path):

        img = pil_image.open(path)

        hw_tuple = (self.img_shape[1], self.img_shape[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)

        x = image.img_to_array(img)[:, :, 0:3]
        x = self.preprocess_input(x)

        return x

    def predict_one(self, image_in):
        x = self.prepare_img(image_in)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        res = self.model.predict(x)
        # {'fish': 0, 'nonfish': 1}
        return 1.0 - res[0][0]

    def predict_mean_one(self, image_in, n_tran=3):
        img_arr = np.zeros((n_tran, *self.img_shape, 3), dtype='float32')

        img = self.prepare_img(image_in)
        for i in range(n_tran):
            if i == 0:
                img_arr[i] = img
                continue

            x = self.train_datagen.random_transform(img)
            img_arr[i] = x

        img_arr = self.preprocess_input(img_arr)
        res = self.model.predict(img_arr)

        return 1.0 - res.mean()

    def predict_vid(self, vid, batch_size=None):

        vid_leng = vid.get_length()
        img_arr = np.zeros((vid_leng, *self.img_shape, 3), dtype='float32')

        crop_area = get_ruler_Crop_Area(vid)
        for i, img in enumerate(vid):
            crop_img = crop_image(img, crop_area)
            img_arr[i] = self.prepare_img(crop_img)

        img_arr = vgg16.preprocess_input(img_arr)
        prediction = self.model.predict(img_arr, batch_size=batch_size)

        return 1.0 - prediction

    def predict_mean_vid(self, vid, batch_size=None, n_tran=3):

        vid_leng = vid.get_length()
        img_arr = np.zeros((vid_leng * n_tran, *self.img_shape, 3), dtype='float32')

        crop_area = get_ruler_Crop_Area(vid)
        for i, img in enumerate(vid):
            crop_img = crop_image(img, crop_area)
            img = self.prepare_img(crop_img)

            for j in range(n_tran):
                if j == 0:
                    img_arr[i * n_tran + j] = img
                    continue

                x = self.train_datagen.random_transform(img)
                img_arr[i * n_tran + j] = x

        img_arr = vgg16.preprocess_input(img_arr)
        prediction = self.model.predict(img_arr, batch_size=batch_size)

        prediction_mean = np.zeros((vid_leng, 1), dtype='float32')
        for i in range(int(vid_leng / n_tran)):
            prediction_mean[i] = prediction[i * n_tran: (i + 1) * n_tran].mean()

        return 1.0 - prediction

    def predict_dir(self, dir_in, path_out):
        df = pd.DataFrame(columns=['video_id', 'frame', 'prob'])
        files = glob.glob(dir_in + '/*.png')

        pbar = tqdm(total=len(files))
        for path in files:
            video_id, frame = os.path.basename(path)[:-4].split('_')
            x = self.load_img(path)
            x = np.expand_dims(x, axis=0)
            res = self.model.predict(x)
            prob = 1.0 - res[0][0]

            df = df.append({'video_id': video_id,
                       'frame': frame,
                       'prob': prob
                       }, ignore_index=True)

            pbar.update(1)

        df.to_csv(path_out, index=False)

    def predict_mean_dir(self, dir_in, path_out, n_tran=3):
        df = pd.DataFrame(columns=['video_id', 'frame', 'prob'])
        files = glob.glob(dir_in + '/*.png')

        pbar = tqdm(total=len(files))
        for path in files:
            video_id, frame = os.path.basename(path)[:-4].split('_')
            x = self.load_img(path)

            img_arr = np.zeros((n_tran, *self.img_shape, 3), dtype='float32')

            for i in range(n_tran):
                # if i == 0:
                #     img_arr[i] = x
                #     # img_arr[i] = self.preprocess_input(x)
                #     continue

                x = self.train_datagen.random_transform(x)
                img_arr[i] = x
                # img_arr[i] = self.preprocess_input(x)

            # img_arr = self.preprocess_input(img_arr)
            res = self.model.predict(img_arr)

            print(res, res.mean())
            prob = 1.0 - res.mean()

            df = df.append({'video_id': video_id,
                       'frame': frame,
                       'prob': prob
                       }, ignore_index=True)

            pbar.update(1)

        df.to_csv(path_out, index=False)





def get_datagens(preprocess_input_fun):
    train_datagen = ImageDataGenerator(
        # shear_range=0.1,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        channel_shift_range=0.01,
        fill_mode='wrap',
        # zoom_range=0.1,
        vertical_flip=True,
        horizontal_flip=True,
        preprocessing_function=preprocess_input_fun)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_fun)

    return train_datagen, test_datagen