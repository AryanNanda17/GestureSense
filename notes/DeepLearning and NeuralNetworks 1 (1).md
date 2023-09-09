# Binary Classification
![](https://hackmd.io/_uploads/ByVIdsNp2.png)
Here x is a feature vector of all pixels detected in a particular image , y is output whether image is a cat or not  
X is a vector containing pixels  of all training examples, while Y is output of all examples  
x<sup>(i)</sup> is the set of i'th training example

# Logistic regression
![](https://hackmd.io/_uploads/rJdD9sNp3.png)
y hat is the probablity of the image being an cat if there is an input of an image. 
We calculate this probablity with the help of a sigmoid function. 
The parameters of logical regression are W , an n<sub>x </sub> dimensional vector  and b, a real number

# Loss and Cost Function
Loss function gives an idea of how much the output is deviated from actual result/ error in output.
![](https://hackmd.io/_uploads/SJxoihiV6n.png)
Loss function is for a single training example.
Cost function is average of the loss function over the entire training set
![](https://hackmd.io/_uploads/HkGmao4Tn.png)

# Gradient Descent

If we plot J(w,b) with w and b in 3D, we have to find a minima in the space such that J is smallest. We use gradient descent method for this.
We will repeatedly apply the following formula : 
![](https://hackmd.io/_uploads/ry2Gl3VT3.png)
and 
![](https://hackmd.io/_uploads/HJsQxn46h.png)
where : means updating the value of w or b as. We will gradually reach the minimum.

# Computational Graph

![](https://hackmd.io/_uploads/HJ6oMWrTh.png)
Here, the blue arrows indicate forward computation or forward propogation while red arrows indicate backward propogation or computation. 
For e.g. in forward propogation, next step is the derivative of previous step

The coding convention dvar represents derivative of final output variable with respect to various intermediate quantities

![](https://hackmd.io/_uploads/rk8jzTB6h.png)
![](https://hackmd.io/_uploads/Bkso7aBT3.png)

# Vectorisation

Performing a for loop to calculate z takes more time
![](https://hackmd.io/_uploads/SkZbSTSa3.png)

Instead we use vectorisation as follows : 
![](https://hackmd.io/_uploads/SkXRBaHp2.png)

![](https://hackmd.io/_uploads/r1QNupB62.png)

![](https://hackmd.io/_uploads/rypSFpSa3.png)

# Broadcasting in Python

Sum of elements of array vertically is axis = 0
"" horizontally is axis = 1

![](https://hackmd.io/_uploads/rJekjTrp3.png)

Broadcasting can be understood as follows : 
![](https://hackmd.io/_uploads/B18mjarp2.png)

![](https://hackmd.io/_uploads/BkzNh6ra3.png)

'*' indicates element wise multiplication, not matrix multiplication, np.dot() indicates matrix multiplication

In a neural network we use superscript [] to denote layer.
Back calculation can be done as follows : 
![](https://hackmd.io/_uploads/rk6D_X8ah.png)

A neural network looks as follows : 
![](https://hackmd.io/_uploads/B1i_tX863.png)
 The activations of hidden layer are represented by the following matrix: 
 ![](https://hackmd.io/_uploads/rJTcFmITn.png)
 
 When there is a hidden layer present, we can vectorise the activation and z as follows : 
 
  
