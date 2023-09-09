# Train/Dev/Test set

![](https://hackmd.io/_uploads/HJZ2ABkA2.png)


We first run the training set and try out on dev set , doing many times thus tuning out model. We make the final test on the test set

Our model generally has 60% data in train set, 20% in dev set and 20% in test set

#### Our train, dev and test sets should be from the same source generally

# Bias and Variance

High bias implies high error / deviation from actual results

High variance implies higher error difference between train set and dev set results

To tackle high bias we can use bigger network while for high variance we can use more data and regularisation

# Regularisation

Regularisation can be done as follows in the cost function :

![](https://hackmd.io/_uploads/BJWWbUkR2.png)

Where frobenius norm is : 
![](https://hackmd.io/_uploads/ByEN-LkA3.png)

Thus the formula of gradients becomes : 
![](https://hackmd.io/_uploads/HJrI-UJA2.png)

Lambda is regularisation parameter

Intuitionally we can say that a higher value of lambda causes the program to reduce the values of weights of the layers. This in way causes the network to move a bitcloser to simpler networks by behaviour, thus removing problem of overfitting

## Dropout regularisation

We randomly remove certain nodes and check the model for its efficiency. 

Inverted dropout can be seen as below : 
![](https://hackmd.io/_uploads/H1Sx0I1R3.png)

We divide a by keep-prob so tht the effective output of z doesnt change

## Augmentation

Flipping a given image horizontally/vertically/ cropping/ zooming in or out can get you more training images without having to get out and get newer images

## Early stopping
![](https://hackmd.io/_uploads/Sk99I4gRn.png)

# Normalisation

It brings input features on similar scales 

# Vanishing/Exploding gradients

If the weight matrix is slightly larger than identity matrix, for very deep neural networks weights become very large thus increasing yhat a lot

Similarly for w matrix slightly smaller than identity matrix, weights become very smaller causing yhat to become very small

This caused hinderance in training large neural networks, however it can be sort of overcome in following ways : 

## Weight Initialisation
 We can initilaise the weight for reLU as 
 ![](https://hackmd.io/_uploads/BJvmBBgRn.png)

for tanh : 
![](https://hackmd.io/_uploads/SJPVSHe0h.png)

for other cases : 
![](https://hackmd.io/_uploads/rJrSSHlAn.png)

## Gradient Checking

Doing a two sided gradient checking improves gradient calculation accuracy

![](https://hackmd.io/_uploads/rkRysBlRn.png)

![](https://hackmd.io/_uploads/r1iTjre0n.png)

### Gradient checking implementation notes

+ Dont use in training only to debug,as d(theta)calculation is a very slow process
+ If algorithm fails try to look at components and find the bug, i.e. find that i'th thetas that are having high errors
+ Remember to include regularisation
+ Doesnt work in case of dropout

# Mini Batch Gradient Descent

While training on very large data sets the computation and gradient descent becomes slow...thus we divide the data into mini batches and perform gradient descent on those individual batches

Notation is as follows:

Too small or too large minibatch size is not preferred. Preferred batch sizes are 64,128,256,512....

![](https://hackmd.io/_uploads/rJFxEtNCh.png)
![](https://hackmd.io/_uploads/SkCM4KVA3.png)

Learning rate decay slowly reduces rate of learning that helps to close in on the actual answer

![](https://hackmd.io/_uploads/B1MVNYNCn.png)
Other decay methods :   
![](https://hackmd.io/_uploads/ryamYY4R3.png)

# Hyperparameter tuning

If we have limited computational resources following hyperparameters must be trained first : learning_rate,batch_size,number of epochs, 

One of the ways is to use a grid to randomyl try out certain combinations of required hyperparameters to see which are suited the best
![](https://hackmd.io/_uploads/BkfW_HwCn.png)

Another way is coarse to fine . If some nearby hyperparameter combinations are working well then we focus on that region itself and find the required combination 
![](https://hackmd.io/_uploads/Hynu_Bw02.png)

Sometimes for eg most probable values can lie in 0.1 to 1 while we are considering 0.0001 to 1.... so we convert this into a logarithmic scale to find the hyperparameters effectively
![](https://hackmd.io/_uploads/Syf1oSw03.png)

There are two approaches to train a model. One where we focus all our attention in training and improving (or babysitting) one model is called as pandas approach. While the other one where we parallely train multiple models to see which performs the best is called as caviars approach

Normalising : 
![](https://hackmd.io/_uploads/SJM03BP02.png)
 It can be done as follows : 
 ![](https://hackmd.io/_uploads/SJnE0HD02.png)
After normalising, we use z as follows : 
![](https://hackmd.io/_uploads/BJCUASD03.png)

Batch normalisation is done on minibatches , as we are calculating mean in a step , using 'b' becomes irrelevant, so we directly use beta in new z  as follows : 
![](https://hackmd.io/_uploads/HyM6E5vR3.png)
![](https://hackmd.io/_uploads/H1qjN5w03.png)
![](https://hackmd.io/_uploads/SksbB9D03.png)

Using a larger minibatch size one reduces the regularisation effect

![](https://hackmd.io/_uploads/ryyQCMdRn.png)

When we have a multiple classifier, we need to predict probablities of various possible outputs, we use the following softmax function for this purpose : 
![](https://hackmd.io/_uploads/HJJu1XOAn.png)
![](https://hackmd.io/_uploads/BJMFk7_03.png)
![](https://hackmd.io/_uploads/Sk8meQOCh.png)

For calculating gradient descent in softmax classifier,last layers backprop is done as follows :
![](https://hackmd.io/_uploads/S1zrZ7uA2.png)

![](https://hackmd.io/_uploads/H1IZGQu02.png)
![](https://hackmd.io/_uploads/ryUIVm_Ch.png)

























