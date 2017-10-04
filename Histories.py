from keras.callbacks import Callback
import pandas as pd


class Histories(Callback):
    def __init__(self, filepath):
        self.filepath = filepath

        super(Callback, self).__init__()

    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

        self.val_acc = []
        self.val_loss = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))

        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))

        df = pd.DataFrame({'acc': self.acc,
                           'loss': self.loss,
                           'val_acc': self.val_acc,
                           'val_loss': self.val_loss})

        df.to_csv(self.filepath)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
