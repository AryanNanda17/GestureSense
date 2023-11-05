<p>
<h1 align = "center" > <strong>GestureSense</strong> <br></h1>

<h3 align = "center">

</p>
:movie_camera: Exploring all possible Deep Learning Models for Webcam-Based Hand <br>Gesture Navigation for Enhanced Accessibility. :clapper:

[SRA](https://www.sravjti.in/) Eklavya 2023<br></h3>

<hr>
<details>
<summary>Table of Contents</summary>

- [Aim](#aim)
- [Description](#description)
- [Tech Stack](#tech-stack)
- [File Structure](#file-structure)
- [Future Scope](#future-scope)
- [Applications](#applications)
- [Project Setup](#project-setup)
- [Usage](#usage)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

</details>

---

## ⭐Aim

* The aim of our project is to develop an intuitive and accessible navigation system that leverages deep learning and computer vision to enable users to control digital devices through natural hand gestures captured by webcams.

---


## 📝Description
We have implemented <strong>3 Deep Learning Models </strong>for <strong>gesture Recognition and Navigation:-<strong>


### 1. Running Averages Model for Bg-Subtraction:-

  We used <strong>VGG7 Architecture</strong> followed by running averages model for bg subtraction for <strong>Gesture Recognition and Navigation</strong>
  We have used OpenCV function "accumulateWeghts" to find the running averages of frames. We manually Created a dataset through which we trained our model.
```python
cv2.accumulateWeighted(src, dst, alpha)
```
   
[This(Running Average model – Background Subtraction)](https://cvexplained.wordpress.com/2020/04/17/running-average-model-background-subtraction/)Article explains the Running Averages Approach for Gesture Recognition very clearly.


### 2. YOLO Hand Gesture Detection Model

We used MobileNet Pretrained Weights from [Tensorflow.zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to implement our Yolo Model. We Manually labelled the Dataset for Hand Detection using [LabelImg](https://github.com/HumanSignal/labelImg). The dataset has 60 images of 4 different gestures(15 each). 

Although we only used 60 images for training our YOLO Model, we got great results in real time. 


### 3. CONV3D+LSTM Model

We created a motion detection model for gesture recognition using <strong>Jester Dataset</strong>. This model consists of 10 CONV3D Layers and 3LSTM layers. 
Here, we extracted SpatioTemporal features for motion detection. We implemented this model in <strong>TensorFlow as well as Pytorch</strong>.
You can refer [this (Attention in Convolutional LSTM for Gesture Recognition)](https://proceedings.neurips.cc/paper_files/paper/2018/file/287e03db1d99e0ec2edb90d079e142f3-Paper.pdf) paper for learning more about Conv3D+LSTM implementation.


---


## 🤖Tech-Stack

#### Programming Language
- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


#### DL Framework

- ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
- ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

#### Image Processing

 - ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)


#### Libraries

- ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
- ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

### File Structure
```  
.
├── 3b1b notes
│   ├── Deep Learning
│   └── Linear Algebra
├── Coursera Notes
│   ├── Course_1 Neural Networks and Deep Learning (Coursera)
│   ├── Course_2 Improving Deep Neural Networks
│   └── Course_4 Convolutional Neural Networks
├── Create_Dataset
│   ├── PreProcessingData.py
│   └── detect.py
├── GestureDetection
│   └── BgEliminationAndMotionDetection.py
├── Hand Detection Using OpenCV
│   ├── Background_subtractor_hand_detection.py
│   └── Skin_Segmentation.py
├── Keras_Models
│   ├── 3DCNN_LSTM.ipynb
│   ├── 3DCNN_LSTM_Pytorch.ipynb
│   ├── GestureWiseMaverick_Masking.ipynb
│   ├── GestureWiseMaverick_NoMasking.ipynb
│   └── Yolo_MobileNet.ipynb
├── Mnist From Scratch Using Jax and Numpy
│   ├── JAX_4L_Autodiff_MNIST_IMPLEMENTATION.ipynb
│   ├── JAX_4L_Without_Autodiff.ipynb
│   ├── NumPy_2L.ipynb
│   └── NumPy_4L.ipynb
├── README.md
├── ResNet-34
│   ├── Assets
│   ├── ResNets_34.ipynb
│   └── Residual model paper.pdf
├── Saved_Models
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
└── environment.yml


```

## Getting Started

### Prerequisites
1. Linux 18.04 or above
2. [TensorFlow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
3. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installed on system




## Contributors

* [Aryan Nanda](https://github.com/AryanNanda17) - nandaaryan823@gmail.com

* [Mihir Gore](https://github.com/MihirGore23) - 

  

## Acknowledgements 
- [SRA Vjti](https://www.sravjti.in/) Eklavya 2023
  
A special thanks to our mentors for this project:
- [Advait Dhamorikar](https://github.com/advait-0) 
- [LakshayaSinghal](https://github.com/LakshayaSinghal)
- [Khushi-Balia](https://github.com/Khushi-Balia)
  <br/>

# License
The [License](LICENSE) used in this project.

---
