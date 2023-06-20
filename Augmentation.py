
from skimage import io
import random
import glob
import albumentations as A


num_of_images= 12000

path = "datasetF"
folder = "augmented data"
images = []
masks = []

for im in sorted(glob.glob(path + '/*img.png')):
    images.append(im)

for msk in sorted(glob.glob(path + '/*mask.png')):
    masks.append(msk)

aug = A.Compose([
    A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.3),
    A.Blur(blur_limit=7, always_apply=False, p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5), # histo equalizer
    A.Downscale(scale_min=0.6, scale_max=0.9, always_apply=False, p=0.3),
    A.HorizontalFlip(p=.25),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.7), #Randomly changes the brightness, contrast, and saturation of an image.
    A.GridDistortion(p=.7)
]
)





i = 0
while i <= num_of_images:
    number = random.randint(0, len(images) - 1)  # PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    print(image, mask)
    # image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)

    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

    io.imsave(f"{folder}/{i}_image.png", transformed_image)
    io.imsave(f"{folder}/{i}_mask.png", transformed_mask)
    i = i + 1