from PIL import Image, ImageFilter
import os
import csv
import numpy as np
from torchvision import transforms


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a jpg file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.LANCZOS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.LANCZOS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values
    print(len(tv))
    return tv


# Path to the folder containing the images
folder_path = r'C:/Users/vorob/OneDrive/Рабочий стол/Alpha\НадрадкГ'


for image_file in os.listdir(folder_path):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(folder_path, image_file)
        resized_image = imageprepare(image_path)
        images_data = np.array(resized_image)

    with open("Gg.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(images_data)

