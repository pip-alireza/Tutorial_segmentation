import glob
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as tf
import segmentation_models as sm
from matplotlib import pyplot as plt


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

path = "test_data"
# careful. it may give incompatible shape error if using different shapes

X_pixel = 512
Y_pixel = 512
train_img = []
train_msk = []

for im in sorted(glob.glob(path + '/*p.png')): # it searches for any file with tag "p"
    image = cv2.imread(im, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (X_pixel, Y_pixel))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    train_img.append(image)

train_images = np.array(train_img)

for msk in sorted(glob.glob(path + '/*m.png')): # it searches for any file with tag "m"
    mask = cv2.imread(msk, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (X_pixel, Y_pixel))
    # Next 2 lines are necessary since we want the value be btw 0 and 1. Therefore we can calculate IOU
    mask[mask > 0] = 255 
    mask = mask/255 
    mask = mask.astype(np.uint8)
    mask = mask[:, :, 0] # We reduce the channel only to one
    train_msk.append(mask)

train_masks = np.array(train_msk)
train_masks=np.expand_dims(train_masks,axis=3)


x_train, x_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.3)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=1,v
    epochs=20,
    shuffle=True,
    validation_data=(x_val, y_val),
)

accuracy = model.evaluate(x_val, y_val)
print(accuracy)
model.save('aorta_unet1.h5')

# Testing the model and displaying the results

pred = model.predict(np.expand_dims(train_images[2, :,:,:], axis=0))  #shape_required (1,X,X,1)
pred_mk = np.squeeze(pred) 
plt.imshow(pred_mk)
plt.show()



