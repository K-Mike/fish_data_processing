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
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from PIL import Image as pil_image
from .utils import get_ruler_Crop_Area, Crop_Area, crop_image, get_class_sim, crop_areas
from .Histories import Histories


class bottelneck_VGG16():

    def __init__(self, model_name=None, verbose=0):
        self.img_shape = (224, 224)

        if model_name is None:
            self.model_name = 'ruler_detection_bottelneck_vgg16'
        else:
            self.model_name = model_name
        # this is the augmentation configuration we will use for training
        self.train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            channel_shift_range=0.1,
            fill_mode='reflect',
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True,
            preprocessing_function=vgg16.preprocess_input)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        self.test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

        # create the base pre-trained model
        base_model = vgg16.VGG16(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

        if verbose > 0:
            print(self.model.summary())

    def load_model(self, path=None):

        if path is None:
            self.model = load_model(self.model_name + '.h5')
        else:
            self.model = load_model(path)

    def fit_generator(self, data_train_dir, data_valid_dir, batch_size, epochs, steps_per_epoch, validation_steps):
        # create data generators
        self.train_generator = self.train_datagen.flow_from_directory(
            data_train_dir,
            target_size=self.img_shape,
            batch_size=batch_size,
            class_mode='binary')

        # this is a similar generator, for validation data
        self.validation_generator = self.test_datagen.flow_from_directory(
            data_valid_dir,
            target_size=self.img_shape,
            batch_size=batch_size,
            class_mode='binary')

        # create callbacks
        log_history = Histories(self.model_name + '.csv')
        checkpointer = ModelCheckpoint(filepath=self.model_name + '.h5', verbose=1, save_best_only=True, monitor='val_acc')

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

    def predict_one(self, image_in):
        x = self.prepare_img(image_in)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)

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

        img_arr = vgg16.preprocess_input(img_arr)
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

    def predict_vid_fast(self, vid, batch_size=None):
        df_train_cl = pd.read_csv('/mnt/data/jupyter/ddata_fish/data_train_hash.csv', index_col=0)

        vid_leng = vid.get_length()
        img_arr = np.zeros((vid_leng, *self.img_shape, 3), dtype='float32')

        cl = get_class_sim(df_train_cl, vid.get_data(0))
        crop_area = crop_areas[cl]
        for i, img in enumerate(vid):
            crop_img = crop_image(img, crop_area)
            img_arr[i] = self.prepare_img(crop_img)

        img_arr = vgg16.preprocess_input(img_arr)
        prediction = self.model.predict(img_arr, batch_size=batch_size)

        return 1.0 - prediction