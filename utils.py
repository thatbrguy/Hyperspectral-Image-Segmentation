import numpy as np
from scipy.io import loadmat
from collections import Counter
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def mat2array():

    data_mat = loadmat('Indian_pines_corrected.mat')
    gt_mat = loadmat('Indian_pines_gt.mat')

    data = data_mat['indian_pines_corrected'].astype(np.float32)
    gt = gt_mat['indian_pines_gt']

    for i in range(data.shape[-1]):
      data[:,:,i] = (data[:,:,i] - np.mean(data[:,:,i])) / np.std(data[:,:,i])

    return data, gt

def load_data():

    data, gt = mat2array()
    train_patches, train_patches_gt = load_patches(data, gt)
    train_patches_synthetic, train_patches_gt_synthetic = load_synthetic_patches(data, gt)

    train_patches = np.concatenate((train_patches, train_patches_synthetic), axis=0)
    train_patches_gt = np.concatenate((train_patches_gt, train_patches_gt_synthetic), axis=0)

    trainX, valX, trainY, valY = train_test_split(train_patches, train_patches_gt, test_size=0.25, random_state=42)
    instances = dict(Counter(list(np.argmax(trainY, axis=3).ravel())))

    total = sum(instances.values())
    weights = np.log10([total/instances[i] for i in range(16)])
    weights_dict = dict([(i, j) for i,j in enumerate(weights)])

    return trainX, valX, trainY, valY, weights_dict

def load_patches(data, gt, patch_size = 4):

    patches = []
    patches_gt = []

    for i in range(data.shape[0] - patch_size):
      for j in range(data.shape[1] - patch_size):
          patch = (data[i:i+patch_size, j:j+patch_size, :]).copy()
          patch_gt = (gt[i:i+patch_size, j:j+patch_size]).copy()
          if np.any(patch_gt == 0):
              continue
          else:
              patches.append(patch)
              patches_gt.append(patch_gt)

    patches_1 = np.array(patches)
    patches_gt = np.array(patches_gt) - 1
    patches_gt_1 = to_categorical(patches_gt, num_classes = 16)

    return patches_1, patches_gt_1

def load_synthetic_patches(data, gt, patch_size = 4, small_patch_size = 1, oversample = 12, label_choice = 8):

    patches_small = [[] for _ in range(16)] #16 classes
    patches_gt_small = [[] for _ in range(16)]

    for i in range(data.shape[0] - small_patch_size):
      for j in range(data.shape[1] - small_patch_size):
          patch = (data[i:i+small_patch_size, j:j+small_patch_size, :]).copy()
          patch_gt = (gt[i:i+small_patch_size, j:j+small_patch_size]).copy()
          if np.any(patch_gt == 0):
              continue
          else:
              index = patch_gt[0,0] - 1
              patches_small[index].append(patch)
              patches_gt_small[index].append(patch_gt)

    patches_small = [np.array(patches_small[i]) for i in range(16)]
    patches_gt_small = [(np.array(patches_gt_small[i]) - 1) for i in range(16)]

    ## Mixed patches

    patches = []
    patches_gt = []

    for sample in range(int(oversample)):
      new_patch = np.zeros((patch_size, patch_size, 200))
      new_patch_gt = np.zeros((patch_size, patch_size))
              
      for i in range(0, patch_size, small_patch_size):
          for j in range(0, patch_size, small_patch_size):
          
              index_choice = np.random.randint(int(len(patches_small[label_choice]) * 0.75))
              new_patch[i:i+small_patch_size, j:j+small_patch_size, :] = patches_small[label_choice][index_choice]
              new_patch_gt[i:i+small_patch_size, j:j+small_patch_size] = patches_gt_small[label_choice][index_choice]            

      patches.append(new_patch)
      patches_gt.append(new_patch_gt)

    patches = np.array(patches)
    patches_gt = np.array(patches_gt)
    patches_gt = to_categorical(patches_gt, num_classes=16)

    return patches, patches_gt
