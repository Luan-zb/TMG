import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

# the GAP is implemented by tensorflow
a=torch.arange(0,24).view(2, 3, 2,2)
a=a.float()
a=a.numpy()
print("This is x: ",a)
y = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(a)
print(y,y.shape)
sess=tf.compat.v1.InteractiveSession()
print(y.eval())



# the GAP is implemented by torch
b=torch.arange(0,24).view(2, 3, 2,2)
print("This is input: ",b)
b=b.float()
b=b.mean(dim=-1)
b=b.mean(dim=-1)
print("This is output: ",b,b.shape)
