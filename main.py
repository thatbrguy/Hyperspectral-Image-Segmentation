import os
import pickle
import argparse
import numpy as np
from collections import Counter
from utils import load_data, mat2array
from sklearn.metrics import cohen_kappa_score as kappa

import keras.backend as K
from keras.backend import clear_session
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2DTranspose, Add, Input, Concatenate, Layer, SeparableConv2D
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU, Conv2D, Reshape

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
parser.add_argument('--batch_size', type=int, help='Batch size', default=200)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
parser.add_argument('--model', help='Choose between models "A" and "B"', choices=['A', 'B'], required=True)
parser.add_argument('--mode', help='Choose between modes "train" and "infer"', choices=['train', 'infer'], required=True)
parser.add_argument('--weights', help='Path of the weights file for inference mode', default='')

class statsLogger(Callback):
    """
    Saving loss and accuracy details to an array
    """
    def __init__(self):
        self.logs = []
    def on_epoch_end(self, epoch, logs):
        logs['epoch'] = epoch
        self.logs.append(logs)

class PixelSoftmax(Layer):
    """
    Pixelwise Softmax for Semantic Segmentation. Also known as
    4D Softmax in some sources. Applies Softmax along the last
    axis (-1 axis). 
    """
    def __init__(self, axis=-1, **kwargs):
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
    """
    U-Net model using vanilla convolutions for downsampling.
    Uses patch_size = 4.
    """
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
    """
    U-Net model using depthwise convolutions for downsampling.
    Uses patch_size = 4.
    """
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

def build_model(model, input_shape):
    """
    Builds the model based on your choice of the mode
    variable.
    """
    clear_session()

    img = Input(shape = input_shape)

    if model == 'A':
        x = separable_model(img)
    elif model == 'B':
        x = conv_model(img)

    x = PixelSoftmax(axis=-1)(x)
    model = Model(inputs=img, outputs=x)

    return model

def train(args):

    trainX, valX, trainY, valY, weights_dict = load_data()

    model = build_model(args.model, input_shape = trainX.shape[1:])

    """
    # Converting shape of trainY, valY from [-1, 4, 4, 16] to [-1, 16, 16]
    # That is, reshaping [None, H, W, C] to [None, H * W, C]
    # This is done to weight the loss function classwise using "sample_weights"
    # Refer: https://github.com/keras-team/keras/issues/3653
    """

    sample_weights = []
    valY = np.reshape(valY, (-1, 16, 16))
    trainY = np.reshape(trainY, (-1, 16, 16))

    """
    # Applying sample_weights as class_weights are not supported for 3+ dimensional
    # targets. We provide a weight value from the weights_dict to each pixel of the
    # 4x4 patch. As explained above, the 4x4 patch is reshaped to a 16-D vector. We
    # pass a 16-D weight_vector to provide pixel-wise loss function weightage, for
    # each training sample. The set of 16-D weight vectors for all training samples
    # is contained in sample_weights.  
    """

    for i in range(len(trainY)):
      patch = trainY[i]
      weight_vector = []
      for j in range(len(patch)):
        weight_vector.append(weights_dict[patch[j].argmax()])
      sample_weights.append(weight_vector)

    sample_weights = np.array(sample_weights)

    history = statsLogger()
    opt = Adam(lr = args.lr, decay = 1e-4)
    ckpt = ModelCheckpoint(filepath = args.model + '.hdf5', 
                           save_best_only = True, 
                           verbose = 1, 
                           monitor = 'val_loss')
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = opt, 
                  metrics = ['accuracy'], 
                  sample_weight_mode = "temporal")
    
    model.fit(trainX, 
              trainY, 
              batch_size = args.batch_size, 
              epochs = args.epochs, 
              validation_data = (valX, valY), 
              callbacks = [ckpt, history], 
              sample_weight = sample_weights)

    # Dumping logs into a pickle file
    with open(args.model + '_logs.pickle', 'wb') as file:
        pickle.dump(history.logs, file, protocol=pickle.HIGHEST_PROTOCOL)

def inference(args):

    data, gt = mat2array()

    patch_size = 4
    output = np.zeros((145, 145, 16))

    clear_session()
    model = build_model(args.model, input_shape = (4, 4, 200))
    model.load_weights(args.weights)

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
    gt_without_test_label, output_without_test_label = [], []
    classwise_prediction_frequency = [0 for _ in range(16)]

    for i in range(145):
        for j in range(145):
            if gt[i,j] != 0:
                total += 1
                if gt[i,j] == masked_output[i,j] :
                    count += 1
                    classwise_prediction_frequency[gt[i,j] - 1] += 1

    for i, j in zip(gt.ravel(), masked_output.ravel()):
        if i != 0:
            gt_without_test_label.append(i)
            output_without_test_label.append(j)

    classwise_gt_frequency = [i for i in Counter(gt_without_test_label).values()]

    classwise_gt_frequency = np.array(classwise_gt_frequency)
    classwise_prediction_frequency = np.array(classwise_prediction_frequency)
    
    classwise_average_accuracy = classwise_prediction_frequency / classwise_gt_frequency

    overall_acc = count / total
    average_acc = np.mean(classwise_average_accuracy)
    kappa_score = kappa(gt_without_test_label, output_without_test_label)

    formatted_string = '\nOverall Accuracy = %.3f \nAverage Accuracy = %.3f \nKappa Score = %.3f \n'
    print(formatted_string % (overall_acc * 100, average_acc * 100, kappa_score * 100))


if __name__ == '__main__':

    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    if args.mode == 'infer':
        if os.path.exists(args.weights):
            inference(args)
        else:
            print('Please specify a valid path for the weights file if you want to use ', end="")
            print('inference mode. You can do this by setting the --weights argument')
