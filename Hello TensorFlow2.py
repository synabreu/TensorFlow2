
# coding: utf-8

# # Hello TensorFlow 2.0 - Your First Program

# 
# 'Hello, World' program is known for a beginner who writes the first coding. Like 'Hello, World', I make first TensorFlow 2.0 program in order to explain how TensorFlow 2.0 works is like this. It is called 'Hello, TensorFlow'
# 
# In the case of creating neural networks, the sample I like to use is one where it learns the relationship between two numbers. So, for example, if you were writing code for a function like this, you already know the 'rules' -- 
# 
# 
# ```
# float hw_function(float x){
#     float y = (2 * x) - 1;
#     return y;
# }
# ```
# 
# So how would you train a neural network to do the equivalent task? Using data! By feeding it with a set of Xs, and a set of Ys, it should be able to figure out the relationship between them. so let's step through it piece by piece.
# 

# ## Install
# 
# Let's start with installing TensorFlow 2.0. Here we are installing TensorFlow and calling '!' as executing commnad environment(cmd) on Jupyter Notebook.
# 
# When you have no GPU on the local computer, you should run this command: !pip install tensorflow==2.0.0-alpha0.
# 
# But if you have a GPU on the local computer, you should run this command: !pip install tensorflow-gpu==2.0.0-alpha0.
# 
# Note that I comment on the situation of using GPU environment. 

# In[2]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0 # if you have no GPU on the local computer')
# !pip install tensorflow-gpu==2.0.0-alpha0 # if you have GPU on the local computer 


# ## Import
# 
# Let me import TensorFlow and calling it tf for ease of use. We then import a library called numpy, which helps us to represent our data as lists easily and quickly.
# 
# The framework for defining a neural network as a set of Sequential layers is called keras, so we import that too. In addition, confirm on the installed TensorFlow version.  

# In[3]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
print(tf.__version__) # check the TensorFlow version out


# ## Define and Compile the Neural Network
# 
# Next we will create the simplest possible neural network. It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.

# In[4]:


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# Now we compile our Neural Network. So we have to specify 2 functions, a loss and an optimizer.
# 
# If you've seen lots of math for machine learning, here's where it's usually used, but in this case it's nicely encapsulated in functions for you. We alredy know that in our function, the relationship between the numbers is y=2x-1. 
# 
# When the computer is trying to 'learn' that, it makes a guess what it is maybe y=10x+10. The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.
# 
# It then uses the OPTIMIZER function to make another guess. Based on how the loss function went, it will try to minimize the loss. At that point maybe it will come up with somehting like y=5x+5, which, while still pretty bad, is closer to the correct result (i.e. the loss is lower). It will repeat this for the number of EPOCHS which you will see shortly. 
# 
# But first, we tell it to use 'MEAN SQUARED ERROR' for the loss and 'STOCHASTIC GRADIENT DESCENT' for the optimizer. Over time you will learn the different and appropriate loss and optimizer functions for different scenarios. 
# 

# In[5]:


model.compile(optimizer='sgd', loss='mean_squared_error')


# ## Providing the Data
# 
# Okay! we'll feed in some data. In this case, we are taking 6 xs and 6 ys. You can see that the relationship between these is that y=2x-1, so where x = -1, y=-3 etc. on and on.  
# 
# A python library called 'Numpy' provides lots of array type data structures that are a defacto standard way of doing it. We declare that we want to use these by specifying the values asn an np.array[]

# In[6]:


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# # Training the Neural Network

# The process of training the neural network, where it 'learns' the relationship between the Xs and Ys is in the **model.fit**  call. This is where it will go through the loop we spoke about above, making a guess, measuring how good or bad it is (aka the loss), using the opimizer to make another guess etc. 
# 
# Next up, it will do it for the number of epochs you specify. When you run this code, you'll see the loss on the right hand side.

# In[7]:


model.fit(xs, ys, epochs=500)


# Finally, you have a model that has been trained to learn the relationshop between X and Y. You can use the **model.predict** method to have it figure out the Y for a previously unknown X. For example, if X = 10, what do you think Y will be? 
# 
# Take a guess before you run this code:

# In[8]:


print(model.predict([10.0]))


# 19, right? But it ended up being a little under. The value of model prediction is about 16 or 17.
# I mean this is not correct value. Why?
# 
# Remember that neural networks deal with probabilities, so given the data that we fed the NN with, it calculated that there is a very high probability that the relationship between X and Y is Y=2X-1, but with only 6 data points we can't know for sure. As a result, the result for 10 is very close to 19, but not necessarily 19. 
# 
# As you work with neural networks, you'll see this pattern recurring. You will almost always deal with probabilities, not certainties. So we will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.
# 
