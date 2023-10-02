from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pandas as pd
import Evaluation
import random
import numpy as np


def set_seed(seed):
    """
    Sets the seed for random, numpy and tensorflow
    :param seed: seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_seed(seed)


def train_model(model, dataset, param, epochs, verbose=0, optimizer=None):
    """
    Train the model on a dataset
    :param model: given model, must have get_var_matrix() method
    :param dataset: DataSet object
    :param param: parameters of data generation
    :param epochs: number of epochs
    :param verbose: verbose level
    :param optimizer: optimizer to use, defaults to SGD(clipnorm=1.0, learning_rate=0.01)
    :return: trained model, metrics
    """
    if optimizer is None:
        optimizer = SGD(clipnorm=1.0, learning_rate=0.01)
    @tf.function
    def get_gradient(model, batch_X, batch_y):
        with tf.GradientTape() as tape:
            # Compute predictions for this batch
            y_pred = model(batch_X)
            # Compute loss for this batch
            loss_value = tf.keras.losses.mean_squared_error(batch_y, y_pred)
            loss_value = tf.reduce_mean(loss_value)

        gradient = tape.gradient(loss_value, model.trainable_variables)
        return gradient

    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, parameters, dataset):
            self.model = model
            self.parameters = parameters
            self.dataset = dataset

            self.gradients = []
            self.metrics = {'metrics': [], 'gradients': []}

        def on_train_batch_begin(self, batch, logs=None):
            self.gradients.append(get_gradient(self.model, self.dataset.X_batch[batch], self.dataset.y_batch[batch][:,0]))

        def on_epoch_end(self, epoch, logs=None):
            matrix_sim = Evaluation.matrix_rmse(self.model.get_var_matrix(),
                                                               self.parameters['companion_matrix'][:param['m']])
            matrix_sim['test_rmse'] = Evaluation.root_mean_squared_error(self.dataset.y_test[:, 0],
                                                                         self.model.predict(self.dataset.X_test, verbose=0)[0])
            matrix_sim['epoch'] = epoch

            for i in range(0, len(self.gradients)):
                temp = []
                for grad in self.gradients[i]:
                    if grad is None:
                        temp.append(0)
                    else:
                        temp.append(grad.numpy().mean())
                self.gradients[i] = temp
            gradients = pd.DataFrame(self.gradients, columns=[var.name for var in model.trainable_variables])
            gradients = gradients.describe().T[['mean', 'std', 'max', 'min']].reset_index().rename(
                columns={'index': 'variable'})
            gradients['epoch'] = epoch

            self.metrics['metrics'].append(matrix_sim)
            self.metrics['gradients'].append(gradients)
            self.gradients = []

    # Prepare model
    model.build((None, dataset.X_train.shape[1],dataset.X_train.shape[2]))
    model.compile(optimizer=optimizer, loss='mse')
    callback = MyCallback(model, param, dataset)
    history = model.fit(dataset.X_train, dataset.y_train[:, 0], validation_data=(dataset.X_val, dataset.y_val[:,0]), epochs=epochs,
                        callbacks=[callback], verbose=verbose, batch_size=dataset.batch_size)
    # Save metrics
    metrics = pd.DataFrame(callback.metrics['metrics'])
    metrics['loss'] = history.history['loss']
    metrics['val_loss'] = history.history['val_loss']
    return model, {'metrics': metrics, 'gradients': pd.concat(callback.metrics['gradients'])}