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
- [Dataset](#dataset)
- [Applications](#applications)
- [Results](results)
- [PreRequisites](#getting-started)
- [Project Setup](project-setup)
- [Future Scope](#future-scope)
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

<img width="480" alt="Screenshot 2023-11-06 at 2 36 34 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/f1146a75-003b-42ee-9bc5-73a3a1445ce2">

  We used <strong>VGG7 Architecture</strong> followed by running averages model for BG subtraction for <strong>Gesture Recognition and Navigation</strong>
  We have used the OpenCV function "accumulateWeghts" to find the running averages of frames. We manually created a dataset through which we trained our model.
```python
cv2.accumulateWeighted(src, dst, alpha)
```
<img width="640" alt="Screenshot 2023-11-06 at 2 33 57 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/efcfa5f2-4095-40ea-a9a8-8f92a41fe000">

<strong>This is the motion detection class that we used in our project.</strong>   

[This(Running Average model – Background Subtraction)](https://cvexplained.wordpress.com/2020/04/17/running-average-model-background-subtraction/)Article explains the Running Averages Approach for Gesture Recognition very clearly.

> <strong>  We have created our own dataset to implement this model.</strong>

### 2. YOLO Hand Gesture Detection Model

<img width="599" alt="Screenshot 2023-11-06 at 2 43 16 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/ea5cc999-94a0-4d81-a6c7-8dceec15cfc2">



We used MobileNet Pretrained Weights from [Tensorflow.zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to implement our Yolo Model. We Manually labeled the Dataset for Hand Detection using [LabelImg](https://github.com/HumanSignal/labelImg). The dataset has 60 images of 4 different gestures(15 each). 

<img width="1262" alt="Screenshot 2023-11-06 at 2 40 31 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/24e6008d-6d4f-46de-94f9-1da90bb585a3">



> <strong>Although we only used 60 images for training our YOLO Model, we got great results in real-time. </strong>


### 3. CONV3D+LSTM Model

<img width="532" alt="Screenshot 2023-11-06 at 2 46 47 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/202de020-d0a6-46b1-ab25-dba69870a781">



We created a motion detection model for gesture recognition using <strong>Jester Dataset</strong>. This model consists of 10 CONV3D Layers and 3LSTM layers. 
Here, we extracted spatio-temporal features for motion detection. We implemented this model in <strong>TensorFlow as well as Pytorch</strong>.
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

### 📁File Structure
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
├── MNIST From Scratch Using Jax and Numpy
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
---

## 📓Dataset
For the <strong>Running Averages Model</strong> approach, we created our own dataset which consists of <strong>14,000 images of 11 different hand gestures.</strong>
We have uploaded our dataset on [Kaggle](https://www.kaggle.com/datasets/aryannanda17/masked-gesture-dataset) with a sample notebook.


---

## Results

### Results of Running Averages BgSubtraction Model With VGG-7 Architecture

https://github.com/AryanNanda17/GestureSense/assets/125150482/e79ebfe2-e972-4c52-88f7-e85c02eaa92b

<br>


### Results with YOLO Object Detection Model with MobileNet Pretrained Weights(Consisting of only 60 Images)

https://github.com/AryanNanda17/GestureSense/assets/125150482/8081f056-929d-4cb3-b4ed-4ef81baca067


---
<br>
<br>

## 💸Applications




https://github.com/AryanNanda17/GestureSense/assets/125150482/ecf19b42-a499-4059-aabc-235ec065ebf1





- <strong>Gaming:</strong> Implement gesture-based controls in gaming applications to provide a more immersive and interactive gaming experience, allowing players to control in-game actions through hand movements.

- <strong>Accessibility Tools:</strong> The project has the potential to create accessibility tools that empower individuals with disabilities to control computers, mobile devices, and applications using hand gestures, enhancing their digital independence.

- <strong>Educational Platforms:</strong> The project could lead to the development of interactive educational platforms where teachers and students can engage with digital content, presentations, and simulations using gestures, fostering more engaging and immersive learning experiences.

- <strong>Human-Robot Interaction:</strong> The project has the potential to improve human-robot interactions by enabling robots to understand and respond to human gestures, making collaborative tasks more intuitive and efficient.
  
---


## 🛠Getting Started

### Prerequisites
1. Linux 18.04 or above
2. [TensorFlow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
3. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installed on system


## Project Setup
## 🛠Project Setup

Start by cloning the repo in a directory of your choice
> `git clone https://github.com/AryanNanda17/GestureSense`
> 
Navigate to the Project directory
Let's say you cloned in your Desktop
>  cd Desktop/GestureSense
> 
Create a virtual environment with the required dependencies
>
>`conda env create --name envname -f environment.yml`
>
Switch environment
>
>`conda activate envname`
>
Create a Dataset of Your choice
>cd Create_Dataset
>
Here you can add path where you want the dataset to get collected and the name of your Gesture Label

To do that run:-
>python3 detect.py -h
>
<img width="693" alt="Screenshot 2023-11-06 at 3 40 49 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/c18a4166-63f2-449c-b7f7-d4cc06120d05">


<br>
for example :

- choose an Image path:
  
    > python3 main.py -p "Your Path Here"           
    

-  choose Label : 
    > python3 main.py -l GestureName

Now, we will parse this dataset through another code for masking. For that run:-
>python3 PreProcessingData.py "Image_Path"
>

<img width="679" alt="Screenshot 2023-11-06 at 3 50 21 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/1e24144e-597e-4ee5-af28-80e6939b03a2">

<br>
<br>

Now, you can train your own model with the selected gestures using the [VGG-7 Architecture](https://github.com/AryanNanda17/GestureSense/blob/master/Keras_Models/GestureWiseMaverick_Masking.ipynb)

After Training Export your Model

Run the Running Averages Bg_Subtraction Model

For that First Navigate there:-
>cd ~/Desktop/GestureSense/GestureDetection
>
Now, Run the following command

>python your_script.py -m /path/to/your/model -c 1finger 2finger 3finger C ThumbRight fingersclosein italydown kitli pinky spreadoutpalm yoyo
>
Hence the setup of the Running Averages BgSubtraction Model is completed.

<img width="858" alt="Screenshot 2023-11-06 at 3 58 43 AM" src="https://github.com/AryanNanda17/GestureSense/assets/125150482/30109e69-12c0-4e23-8f65-e61b1b40ebd0">

---

## 🔮Future Scope

- Attention Mechanism Integration: Incorporate attention mechanisms into the CONV3D+LSTM model to improve its ability to focus on relevant features in gesture sequences, enhancing accuracy.
- Mouse Control with YOLO Object Detection API: Implement mouse control functionality using the YOLO object detection API, allowing users to manipulate their computers using gesture-based control with high accuracy.
- Interactive Web Platform Development: Create an interactive web platform that provides users with a user-friendly interface to access and utilize the gesture control system. This platform should be compatible with various browsers and operating systems.

---

## Contributors

* [Aryan Nanda](https://github.com/AryanNanda17) - nandaaryan823@gmail.com

* [Mihir Gore](https://github.com/MihirGore23) - mihirvgore@outlook.com

  

## Acknowledgements 
- [SRA Vjti](https://www.sravjti.in/) Eklavya 2023
  
A special thanks to our mentors for this project:
- [Advait Dhamorikar](https://github.com/advait-0) 
- [LakshayaSinghal](https://github.com/LakshayaSinghal)
- [Khushi-Balia](https://github.com/Khushi-Balia)
  <br/>

# License
The [LICENSE](LICENSE.txt) used in this project.

---
