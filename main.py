import pickle
import numpy as np
from utils import load_data, mat2array

import keras.backend as K
from keras.backend import clear_session
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2DTranspose, Add, Input, Concatenate, Layer, SeparableConv2D
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU, Conv2D, Reshape


# Indian_pines_corrected.mat
# Indian_pines_gt.mat

class statsLogger(Callback):
    def __init__(self):
        self.logs = []
    def on_epoch_end(self, epoch, logs):
        logs['epoch'] = epoch
        self.logs.append(logs)

class PixelSoftmax(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(PixelSoftmax, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape

def conv_model(img):

    ## Depthwise Size 4
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1_2', use_bias=False)(img)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op1 = x

    x = Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', name='conv2_2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op2 = x

    x = Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', name='conv3_2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op3 = x

    x = Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2), padding='same', name='deconv3', use_bias=False)(op3)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Concatenate(axis=-1)([x, op2])

    x = Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding='same', name='deconv2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Concatenate(axis=-1)([x, op1])

    x = Conv2DTranspose(16, kernel_size=(3,3), strides=(1,1), padding='same', name='deconv1')(x)
    x = Reshape((16, 16))(x)

    return x

def separable_model(img):

    ## Depthwise Size 4
    x = SeparableConv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv1_2', use_bias=False)(img)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op1 = x

    x = SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', name='conv2_2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op2 = x

    x = SeparableConv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', name='conv3_2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    op3 = x

    x = Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2), padding='same', name='deconv3', use_bias=False)(op3)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Concatenate(axis=-1)([x, op2])

    x = Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding='same', name='deconv2', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Concatenate(axis=-1)([x, op1])

    x = Conv2DTranspose(16, kernel_size=(3,3), strides=(1,1), padding='same', name='deconv1')(x)
    x = Reshape((16, 16))(x)

    return x

def build_model(mode, input_shape):

    clear_session()

    img = Input(shape = input_shape)

    if mode == 'conv':
        x = conv_model(img)
    elif mode == 'separable':
        x = separable_model(img)

    x = PixelSoftmax(axis=-1)(x)
    model = Model(inputs=img, outputs=x)

    return model

def train(mode):

    trainX, valX, trainY, valY, weights_dict = load_data()

    model = build_model(mode, input_shape = trainX.shape[1:])

    sample_weights = []

    # Converting shape of trainY, valY from [-1, 4, 4, 16] to [-1, 16, 16]
    # This is done to provide classwise weighted regularization

    valY = np.reshape(valY, (-1, 16, 16))
    trainY = np.reshape(trainY, (-1, 16, 16))

    for i in range(len(trainY)):
      array = trainY[i]
      temp_weight = []
      for j in range(len(array)):
        temp_weight.append(weights_dict[j])
      sample_weights.append(temp_weight)

    sample_weights = np.array(sample_weights)

    history = statsLogger()
    opt = Adam(lr = 0.001, decay = 1e-4)
    ckpt = ModelCheckpoint(filepath = mode + '.hdf5', 
                           save_best_only = True, 
                           verbose = 1, 
                           monitor = 'val_loss')
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy'], 
                  sample_weight_mode="temporal")
    
    model.fit(trainX, 
              trainY, 
              batch_size = 200, 
              epochs = 100, 
              validation_data = (valX, valY), 
              callbacks = [ckpt, history], 
              sample_weight = sample_weights)

    # Dumping logs into a pickle file
    with open(mode + '.pickle', 'wb') as file:
        pickle.dump(history.logs, file, protocol=pickle.HIGHEST_PROTOCOL)

def inference(mode, weights):

    data, gt = mat2array()

    patch_size = 4
    output = np.zeros((145, 145, 16))

    clear_session()
    model = build_model(mode, input_shape = (4, 4, 200))
    model.load_weights(weights)

    for i in range(0, data.shape[0] - patch_size, patch_size):
        for j in range(0, data.shape[1] - patch_size, patch_size):
            patch = (data[i:i+patch_size, j:j+patch_size, :]).copy()
            patch = np.reshape(patch, (-1, patch_size, patch_size, 200))
            output[i:i+patch_size, j:j+patch_size, :] = model.predict(patch).reshape(1,patch_size,patch_size,-1)

    # Semantically segmented output -- making it 1-indexed.
    output_semantic = np.argmax(output, axis=2) + 1

    # Getting regions where ground truth is defined
    gt_mask = np.where(gt > 0, np.ones_like(gt), np.zeros_like(gt))

    # Output after filtering 
    masked_output = (output_semantic * gt_mask)

    count = 0
    total = 0
    for i in range(145):
        for j in range(145):
            if gt[i,j] != 0:
                total += 1
                if gt[i,j] == masked_output[i,j] :
                    count += 1

    accuracy = count / total

    print('Accuracy = %.3f' % (accuracy * 100))


if __name__ == '__main__':

    #train('separable')
    inference('separable', 'separable.hdf5')
