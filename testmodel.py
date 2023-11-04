import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

###Step 7: Present solution
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
unique_labels = np.load('unique_labels.npy')
model = tf.keras.models.load_model('best_handwritten.model')
#Test the final model
y_pred = model.predict(x_test)
y_test = np.argmax(y_test, axis=1) #Convert one-hot encoded labels back to integers
y_pred = np.argmax(y_pred, axis=1)
#Visualize the confusion matrix and accuracy score
confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy}')
plt.show()
#Visualize how the epoch steps affect the loss and accuracy inside the model for training and validation
loss_image_data = cv2.imread("loss_plot.png")
accuracy_image_data = cv2.imread("accuracy_plot.png")
cv2.imshow("Loss Plot", loss_image_data)
cv2.imshow("Accuracy Plot", accuracy_image_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Look at some examples the model failed
misclassified_indices = np.where(y_test != y_pred)[0]
for i in range(3):
    misclassified_index = misclassified_indices[np.random.choice(len(misclassified_indices))]
    image = (x_test[misclassified_index] * 255).astype(np.uint8) #Misclassified image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f'Image {i + 1}', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
#Overall, the model seems to be working great!
#The next step is to draw images ourselves in paint, and test our model based on these.

#See launch.py next