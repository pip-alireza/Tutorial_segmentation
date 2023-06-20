# segmentation_tutorial

This tutorial shows a simple implementation of UNet for medical image segmentation. However, it's important to note:

The test dataset used in the tutorial consists of only few images. If you plan to use the provided code with your own dataset, ensure that you update the file addresses accordingly. Also, update the key words in line 22 and 32. 

With a limited number of epochs (only 70), the model should generate output. Please note, this out is due overfitting
One of the images from the test dataset is utilized for testing purposes. It's crucial to have a separate set of test images that are distinct from the training data to accurately evaluate the model's performance.

To run the program, download the repository and extract the code folder. Make sure all the required libraries mentioned in the tutorial are installed. You can then test the program with the provided test dataset or modify it to train with your own dataset. Remember to adjust the file addresses and prepare your dataset appropriately for the segmentation task.

By following these guidelines, you can proceed with running the UNet implementation for medical image segmentation, either using the provided test dataset or your own data. Adapt the code and parameters as necessary to achieve your desired results.

The images here are found at: https://ddxof.com/ct-interpretation-abdomenpelvis/
I used ITK snap for annotating the aorta


Finally, make sure to install the following libraries
segmentation-models 
Keras 2.3.1
Torch 1.8
Tensorflow 2.4
scikit-learn
 
