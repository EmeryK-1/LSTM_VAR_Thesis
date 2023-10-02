import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class DataSet:
    """
    Preprocessing.
    """
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocessing(self, horizon, back_horizon, train_size=0.7, val_size=0.2, scale=True, batch_size=32):
        self.horizon = horizon
        self.back_horizon = back_horizon

        self.y = self.data.copy().astype('float')
        self.train = self.y[:int(train_size*len(self.y))]
        self.val = self.y[int(train_size*len(self.y))-self.back_horizon:int((train_size+val_size)*len(self.y))]
        self.test = self.y[int((train_size+val_size)*len(self.y))-self.back_horizon:]

        self.scaler = self.scaler.fit(self.train)
        if scale:
            self.val = self.scaler.transform(self.val)
            self.test = self.scaler.transform(self.test)
            self.y = self.scaler.transform(self.y)
        # Training set
        self.X_train, self.y_train = self.create_sequences_multi(self.train,
                                                                 self.train,
                                                                 self.horizon,
                                                                 self.back_horizon)

        # Validation set
        self.X_val, self.y_val = self.create_sequences_multi(self.val,
                                                             self.val,
                                                             self.horizon,
                                                             self.back_horizon)
        # Testing set
        self.X_test, self.y_test = self.create_sequences_multi(self.test,
                                                               self.test,
                                                               self.horizon,
                                                               self.back_horizon)

        # training on all database
        self.X_train_all, self.y_train_all = self.create_sequences_multi(self.y,
                                                                         self.y,
                                                                         self.horizon,
                                                                         self.back_horizon)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.train_dataset = self.train_dataset.batch(batch_size)
        self.batch_size = batch_size
        self.train_dataset = self.train_dataset.shuffle(buffer_size=len(self.data))
        self.X_batch, self.y_batch = [], []
        for batch_x, batch_y in self.train_dataset:
            self.X_batch.append(batch_x.numpy())
            self.y_batch.append(batch_y.numpy())

        return self

    def unscale(self, data):
        return self.scaler.inverse_transform(data)

    @staticmethod
    def create_sequences_multi(X, y, horizon, time_steps):
        Xs, ys = [], []
        for i in range(0, len(X)-time_steps-horizon+1, 1):
            temp = X[i:(i+time_steps), :]
            # Invert order of rows
            temp = temp[::-1]
            Xs.append(temp)
            ys.append(y[(i+time_steps):(i+time_steps+horizon), :])
        return np.array(Xs), np.array(ys)
