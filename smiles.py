#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import random 
np.set_printoptions(threshold=sys.maxsize)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard


#setting the path to the directory containing the pics
smilePath = 'smile/0/'
notsmilePath = 'smile/1'

#appending the pics to training and test data 
X = []
y = []

for img in os.listdir(smilePath):
    if (img != '.DS_Store'):
        pic = cv2.imread(os.path.join(smilePath,img))
        # print(img)
        # print("img")
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic,(64,64))
        X.append(pic)
        y.append(1)

for img in os.listdir(notsmilePath):
    if (img != '.DS_Store'):
        pic = cv2.imread(os.path.join(notsmilePath,img))
        # print(img)
        # print("img")
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic,(64,64))
        X.append(pic)
        y.append(0)


y = np.array(y)
X = np.array(X)
# print("X.shape")
# print(X.shape)
# print(X)

# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(X_train)
# print(y_train)
# print('X_train.shape')
# print('y_train.shape')
# print(X_train.shape)
# print(y_train.shape)


i = random.randint(1,600) # select any random index from 1 to 600
plt.imshow( X_train[i] )
# plt.show()
# print(y_train[i])

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(X_train) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( X_train[index])
    axes[i].set_title(y_train[index], fontsize = 25)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
# plt.show()

X_train = X_train/255
X_test = X_test/255

# print(X_train.shape)
# #(6939, 64, 64, 3)
# print(y_train.shape)
# #(6939,)

cnn_model = Sequential()

cnn_model.add(Conv2D(64, (6, 6), input_shape = (64,64,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(64, (5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Flatten())
cnn_model.add(Dense(activation="relu", units=128))
cnn_model.add(Dense(activation="sigmoid", units=1))

cnn_model.compile(loss ='binary_crossentropy', optimizer=Adam(learning_rate=0.001),metrics =['accuracy'])

epochs = 5

history = cnn_model.fit(X_train,
                        y_train,
                        batch_size = 30,
                        epochs = epochs,
                        verbose = 1)

evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

# get the predictions for the test data
# predicted_classes = cnn_model.predict_classes(X_test)

predictions = (cnn_model.predict(X_test) > 0.5).astype("int32")


# predict_x=model.predict(X_test) 
# classes_x=np.argmax(predict_x,axis=1)

# print(predictions.shape)
# print(y_test.shape)

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Prediction Class = {}\n True Class = {}".format(predictions[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test.T, predictions))

