# Hyperspectral-Image-Segmentation
Semantic Segmentation of HyperSpectral Images using a U-Net with Separable Convolutions.

#### WIP. This repository is incomplete.

## Features
- HyperSpectral Images (HSI) are semantically segmented using two variants of U-Nets and their performance is comparaed.
- **Model A** uses Depthwise Separable Convolutions in the downsampling arm of the U-Net, and **Model B** uses Convolutions in the downsampling arm of the U-Net. 
- Due to the lack of multiple HSI image and ground truth pairs, we train the models by extracting patches of the image. Here, patches are smaller square regions of the image. After training the model, we make predictions patch-wise. 
- Patches are extracted using a stride of 1 for training. We used patches of size `patch_size = 4` (4x4 square regions) for our experiment.
- We use the `sample-weight` feature offered by Keras to weight the classes in the loss function by their log frequency. We use this as there is a skew in the number of examples per class.
- Some classes do not have patches of size `patch_size = 4`. For these classes, we create synthetic patches of size `patch_size = 4` by using patches of size 1. 
- Experimental results are tabulated below.

## Requirements
- Keras
- TensorFlow (>1.4)
- Scikit-Learn
- Numpy
- Scipy

## Instructions
1. Clone the repository and change working directory using:
```
git clone https://github.com/thatbrguy/Hyperspectral-Image-Segmentation.git \
&& cd Hyperspectral-Image-Segmentation
```
2. Download the dataset from [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) **or** by using the following commands:
```
wget http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
wget http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat
```
3. Train the model using:
```
python main.py \
--model "Model A" \
--mode train
```
The file `main.py` supports a few options, which are listed below:
- `--model`: (**required**) Choose between `Model A` and `Model B`.
- `--mode`: (**required**) Choose between training (`train`) and inference (`infer`) modes.
- `--weights`: (**required for inference only**) Path of the weights file for inference mode.
- `--epochs`: Set the number of epochs. Default value is `100`.
- `--batch_size`: Set the batch size. Default value is `200`.
- `--lr`: Set the learning rate. Default value is `0.001`.

## Results

### Indian Pines Dataset

#### Output Maps
|    Ground Truth   	|         Model A        	|         Model B         	|
|:-----------------:	|:----------------------:	|:-----------------------:	|
| ![GT](src/gt.png) 	| ![modelA](src/sep.png) 	| ![modelB](src/conv.png) 	|

#### Epoch vs Accuracy
<img width=70% height=70% src="/src/accfinal.png" alt="Plot"></img>

#### Tabulatation

<table>
  <thead>
    <tr>
      <th rowspan="2">Class Number</th>
      <th rowspan="2">Class Name</th>
      <th rowspan="2">Ground Truth Samples</th>
      <th colspan="2">Model A</th>
      <th colspan="2">Model B</th>
    </tr>
    <tr>
      <td>Predicted Samples</td>
      <td>Accuracy</td>
      <td>Predicted Samples</td>
      <td>Accuracy</td>
    </tr>
   </thead>
   <tbody>
    <tr>
      <td>1</td>
      <td>Alfalfa</td>
      <td>46</td>
      <td>34</td>
      <td>73.91</td>
      <td>31</td>
      <td>67.39</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Corn notill</td>
      <td>1428</td>
      <td>1324</td>
      <td>92.71</td>
      <td>1333</td>
      <td>93.35</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Corn mintill</td>
      <td>830</td>
      <td>759</td>
      <td>91.44</td>
      <td>704</td>
      <td>84.82</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Corn</td>
      <td>237</td>
      <td>189</td>
      <td>79.74</td>
      <td>198</td>
      <td>83.54</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Grass pasture</td>
      <td>483</td>
      <td>448</td>
      <td>92.75</td>
      <td>434</td>
      <td>89.85</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Grass trees</td>
      <td>730</td>
      <td>706</td>
      <td>96.71</td>
      <td>702</td>
      <td>96.16</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Grass pasture mowed</td>
      <td>28</td>
      <td>28</td>
      <td>100</td>
      <td>28</td>
      <td>100</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Hay windrowed</td>
      <td>478</td>
      <td>472</td>
      <td>98.74</td>
      <td>472</td>
      <td>98.74</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Oats</td>
      <td>20</td>
      <td>20</td>
      <td>100</td>
      <td>20</td>
      <td>100</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Soybean notill</td>
      <td>972</td>
      <td>855</td>
      <td>87.96</td>
      <td>850</td>
      <td>87.44</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Soybean mintill</td>
      <td>2455</td>
      <td>2306</td>
      <td>93.93</td>
      <td>2274</td>
      <td>92.63</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Soybean clean</td>
      <td>593</td>
      <td>523</td>
      <td>88.2</td>
      <td>530</td>
      <td>89.38</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Wheat</td>
      <td>205</td>
      <td>175</td>
      <td>85.37</td>
      <td>176</td>
      <td>85.85</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Woods</td>
      <td>1265</td>
      <td>1231</td>
      <td>97.31</td>
      <td>1231</td>
      <td>97.31</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Buildings Grass Trees Drives</td>
      <td>386</td>
      <td>386</td>
      <td>100</td>
      <td>386</td>
      <td>100</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Stone Steel Towers</td>
      <td>93</td>
      <td>89</td>
      <td>95.7</td>
      <td>91</td>
      <td>97.85</td>
    </tr>
    <tr>
      <td colspan="3">Overall Accuracy (OA)</td>
      <td colspan="2">93.13</td>
      <td colspan="2">92.3</td>
    </tr>
    <tr>
      <td colspan="3">Average Accuracy (AA)</td>
      <td colspan="2">92.16</td>
      <td colspan="2">91.52</td>
    </tr>
    <tr>
      <td colspan="3">Kappa Coefficient (K)</td>
      <td colspan="2">92.18</td>
      <td colspan="2">91.24</td>
    </tr>
  </tbody>
</table>

## References:
1. [Dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)
