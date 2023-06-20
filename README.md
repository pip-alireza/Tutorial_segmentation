# segmentation_tutorial

This tutorial shows a simple implementation of UNet for medical image segmentation. Please note:

The test dataset used in the tutorial consists of only few images. If you plan to use the provided code with your own dataset, ensure that you update the file addresses accordingly. Also, update the key words in line 21 and 30. 

With a limited number of epochs (only 50), the model should generate a reasonable output. Please note, this model is overfitting. One of the images from the test dataset is utilized for testing purposes and illustration. To run the program, download the repository and extract the folder. Make sure all the required libraries mentioned below are installed. Remember to adjust the file addresses and prepare your dataset appropriately for the segmentation task.

The test images here are found at: https://ddxof.com/ct-interpretation-abdomenpelvis/
I used ITK snap for annotating the aorta


Required libraries:
segmentation-models 
Keras 2.3.1
Torch 1.8
Tensorflow 2.4
scikit-learn
 
