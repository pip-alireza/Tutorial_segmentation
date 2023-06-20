import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import segmentation_models as sm


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


path = "/test_data"
# careful. it may give incompatible shape error using different shapes
X_pixel = 512
Y_pixel = 512

train_img = []
train_msk = []

for im in sorted(glob.glob(path + '/*img.png')):
    image = cv2.imread(im, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (X_pixel, Y_pixel))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    train_img.append(image)

train_images = np.array(train_img)


for msk in sorted(glob.glob(path + '/*mask.png')):
    mask = cv2.imread(msk, cv2.IMREAD_COLOR)
    mask[mask > 0] = 255 # this turns the image to B&W
    mask = mask/255 # this is necessary for iou since we want the value be btw 0 and 1
    mask = mask.astype(np.float32)
    mask = mask[:, :, 0] # for mask it is 3 dimension image (512,512,3) but all 3 are same
    train_msk.append(mask)

train_masks = np.array(train_msk)
train_masks=np.expand_dims(train_masks,axis=3)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.3)
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)


# binary segmentation (this parameters are default when you call Unet('resnet34')
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')



model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=3, # fro prev aorta_100epoch 256 it was 15
    epochs=40,
    shuffle=True,
    validation_data=(x_val, y_val),
)


accuracy = model.evaluate(x_val, y_val)
print(accuracy)


# Testing the model and displaying the results
pred = model.predict(np.expand_dims(train_images[2, :,:,:], axis=0))  #shape_required (1,X,X,1)
pred_mk = np.squeeze(pred) 
pred_msk = pred_mk > 0.5
plt.imshow(pred_msk)
plt.show()
