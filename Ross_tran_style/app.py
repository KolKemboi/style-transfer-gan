import cv2
import tensorflow as tf 
import keras
import numpy as np
import matplotlib.pyplot as plt

gan = keras.models.load_model("style_gan")

def image_loader(path):
    
    img = tf.keras.utils.load_img(
    path, target_size=(256, 256)
		)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array/255.0

content_image = image_loader("angelina_jolie.jpeg")
style_image = image_loader("ross-tran.png")

outputs = gan(tf.constant(content_image), tf.constant(style_image))

image = np.squeeze(outputs[0])

plt.imshow(image)
plt.show()