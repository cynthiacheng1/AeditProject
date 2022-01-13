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
noAcnePath = 'acne/0/'
acnePath = 'acne/1'

#appending the pics to training and test data 
X = []
y = []

for img in os.listdir(acnePath):
    if (img != '.DS_Store'):
        pic = cv2.imread(os.path.join(acnePath,img))
        # print(img)
        # print("img")
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic,(64,64))
        X.append(pic)
        y.append(1)

for img in os.listdir(noAcnePath):
    if (img != '.DS_Store'):
        pic = cv2.imread(os.path.join(noAcnePath,img))
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

X_train = X_train/255
X_test = X_test/255

print(X_train.shape)
#(6939, 64, 64, 3)
print(y_train.shape)
#(6939,)

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
predictions = (cnn_model.predict(X_test) > 0.5).astype("int32")

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