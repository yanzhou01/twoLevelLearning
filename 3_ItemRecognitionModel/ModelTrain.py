# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import sys 
import os
sys.path.append(os.path.abspath("/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes"))
from fashionnet2 import FashionNet

import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#%matplotlib inline



path = "/Volumes/GoogleDrive/我的云端硬盘/fashion_research/data/labels.pkl"

with open(path, 'rb') as handle:
    temp = pickle.load(handle)

labels = pd.DataFrame(temp)

## choose top and bottom features and get one-hot code
### men from 9940.jpg
men_selected = [
    "トップス",
    "トップスcolor",
    "パンツ",
    "パンツcolor"
]
men_labels = labels.loc[9940:, men_selected]
men_labels = men_labels.iloc[:, [0,1,3,4]]


# one hot encode

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)

## one_code func
def one_code(data):
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    return onehot_encoded
    # invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    #print(inverted)

men_encoded = pd.DataFrame(index = men_labels.index.values, columns=men_selected)
for i in range(men_labels.shape[1]):
    temp = one_code(men_labels.iloc[:,i])
    for j, value in enumerate(temp):
        men_encoded.iloc[j, i] = value






train_image = []
for i in tqdm(range(men_encoded.shape[0])):
    img = image.load_img('/Volumes/GoogleDrive/我的云端硬盘/fashion_research/image_data/coordinate_images/'+str(men_encoded.index.values[i])+'.jpg',target_size=(150,150,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

X = np.array(train_image)



# size=(368,276,3)
image = X
label = men_encoded 

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.3, random_state=10)

y_train_tops = y_train.iloc[:,0]
y_train_tops_color = y_train.iloc[:,1]
y_train_pants = y_train.iloc[:,2]
y_train_pants_color = y_train.iloc[:,3]

y_test_tops = y_test.iloc[:,0]
y_test_tops_color = y_test.iloc[:,1]
y_test_pants = y_test.iloc[:,2]
y_test_pants_color = y_test.iloc[:,3]

# initialize our FashionNet multi-output network (pretrained)
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (150,150,3)

model = FashionNet.build(150,150,
	numCategories_top=y_train_tops.iloc[0,].shape[0],
	numColors_top=y_train_tops_color.iloc[0,].shape[0],
	numCategories_bottom=y_train_pants.iloc[0,].shape[0],
	numColors_bottom=y_train_pants_color.iloc[0,].shape[0],
	finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
	"category_output2": "categorical_crossentropy",
	"color_output2": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0, "category_output2": 1.0, "color_output2": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(X_train,
	{"category_output": np.array(list(y_train_tops)), "color_output": np.array(list(y_train_tops_color)), "category_output2": np.array(list(y_train_pants)), "color_output2": np.array(list(y_train_pants_color))},
	validation_data=(X_test,
		{"category_output": np.array(list(y_test_tops)), "color_output": np.array(list(y_test_tops_color)), "category_output2": np.array(list(y_test_pants)), "color_output2": np.array(list(y_test_pants_color))}),
	epochs=EPOCHS,
	verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("cnn2.h5")

## Predict 


# load model
import tensorflow as tf
from keras.models import load_model
model = load_model('/Volumes/GoogleDrive/我的云端硬盘/fashion_research/cnn2.h5',custom_objects={"tf": tf})
# summarize model.
model.summary()

# example of loading an image with the Keras API
from keras.preprocessing.image import load_img
# load the image
img = load_img('/Volumes/GoogleDrive/我的云端硬盘/fashion_research/image_data/test_img/2.jpg', target_size=(150,150,3))
# report details about the image
#print(type(img))
#print(img.format)
#print(img.mode)
#print(img.size)
# show the image
import matplotlib.pyplot as plt
#plt.imshow(img)

# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
# convert to numpy array
img_array = img_to_array(img)/255
img_array = img_array.reshape((1, 150, 150, 3))
#print(img_array.dtype)
#print(img_array.shape)
# convert back to image
#img_pil = array_to_img(img_array)
#print(type(img))
## predict label
prediction = model.predict(img_array)
idx_top = np.argmax(prediction[0])
idx_top_col = np.argmax(prediction[1])
idx_bottom = np.argmax(prediction[2])
idx_bottom_col = np.argmax(prediction[3])
import pickle

# load the trained convolutional neural network and the label
# binarizer
lb_top = pickle.loads(open("/Volumes/GoogleDrive/我的云端硬盘/fashion_research/men_lb_top", "rb").read())
lb_top_col = pickle.loads(open("/Volumes/GoogleDrive/我的云端硬盘/fashion_research/men_lb_top_col", "rb").read())
lb_bottom = pickle.loads(open("/Volumes/GoogleDrive/我的云端硬盘/fashion_research/men_lb_bottom", "rb").read())
lb_bottom_col = pickle.loads(open("/Volumes/GoogleDrive/我的云端硬盘/fashion_research/men_lb_bottom_col", "rb").read())

print("Top category: %s \nTop color: %s \nBottom category: %s \nBottom color: %s" %(lb_top.classes_[idx_top], lb_top_col.classes_[idx_top_col], lb_bottom.classes_[idx_bottom], lb_bottom_col.classes_[idx_bottom_col]))
