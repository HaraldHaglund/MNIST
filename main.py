import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

### Step 0: Get the data.
mnist = tf.keras.datasets.mnist
#Dataset is split 85%/15% test/train data (as numpy arrays) by default. This distrubution is fairly common and works fine for this application
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

### Step 1: Explore and inspect the data
#Plot three random samples
for i in range(3):
    image = x_train[np.random.choice(len(x_train))]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f'Image {i + 1}', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
#Look at the labels
unique_labels = np.unique(y_test)
for number in unique_labels:
    print('Number ', number ,': ',np.sum(y_test == number))
#Conslusion: We seem to have a very balanced dataset.
#Its also safe to assume that there are no conflicting labels, and that we have IID.

### Step 2: Prepare data, evaluate features
#Note: Dataset is already well prepared. No image reduction needed since the feature space is small already (28x28)
#Lets normalize the grayscale values though (0-255) -> (0-1)
x_train = tf.keras.utils.normalize(x_train,axis =1)
x_test = tf.keras.utils.normalize(x_test,axis =1)
#Reshape to (10000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#Encode labels to a one_hot vector (ie "2" becomes [010000000], "8" becomes [000000010])
y_train = tf.one_hot(y_train.astype(np.int32),depth=10)
y_test = tf.one_hot(y_test.astype(np.int32),depth=10)

### Step 3: Select and create the model
#Since we will be dealing with images i will be using a Convolutional Neural Network (CNN). These are great with images because they use kernels and convolutions to automatically learn and capture patterns of features
model = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
    ]
)
#Print a summary of the model
model.summary()
#Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

###Step 4: Fit and fine-tune the model
#Lets use early stopping stop the training at the right time!
callbacks = [
        EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
        filepath='venv/cp/checkpoint',
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )
]
#We run the model through 10 epochs and choose the model with the minimum loss
history = model.fit(x_train,y_train, epochs=10,validation_data=(x_test,y_test),callbacks=callbacks)
# Create images of the loss and accuracy plots
metrics_df = pd.DataFrame(history.history)
loss_plot = metrics_df[["loss", "val_loss"]].plot()
plt.savefig("loss_plot.png")
plt.close()
accuracy_plot = metrics_df[["accuracy", "val_accuracy"]].plot()
plt.savefig("accuracy_plot.png")
plt.close()
#Save the best model (and some other parameters)
best_model = tf.keras.models.load_model('venv/cp/checkpoint')
best_model.save('best_handwritten.model')
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
np.save('x_test.npy', x_test)
np.save('unique_labels.npy', unique_labels)
#See testmodel.py next