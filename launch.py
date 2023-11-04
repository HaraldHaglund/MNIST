import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

### Step 8: Launch system
#Import the model again
model = tf.keras.models.load_model('best_handwritten.model')
#To compensate for my poor file-structure...
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
#Loop through all images in the "digits" folder. These were made in ms-paint and saved as png
#Predict the number and show the image of the number
image_number = 1
while os.path.isfile(os.path.join(project_dir, f"digits/digit{image_number}.png")):
    try:
        img = cv2.imread(os.path.join(project_dir, f"digits/digit{image_number}.png"))[:, :, 0]
        img = np.invert(np.array(img))  # invert so bg is black instead of white
        norm_img = tf.keras.utils.normalize(img, axis=1)  # normalize again before inserting into the model
        prediction = model.predict(np.array([norm_img]))
        print(f"Predicted number:  {np.argmax(prediction)}")
        # Show the image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f'Image {image_number}', img)
        cv2.waitKey(0)
    except:
        print('Error! File(s) not found')

    finally:
        image_number += 1
#The model got 8/10 correct on my digits imported from paint
#In order to try it for yourself, draw a number on an image of resolution 28x28 and save it as a png file inside the "digits" folder