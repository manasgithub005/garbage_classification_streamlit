import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np


file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
bb = st.button("Submit")

if bb is True:





	model = keras.models.load_model('classifyWaste.h5')


	output_class = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

	test_image = image.load_img(file, target_size = (224,224))
	test_image = image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis=0)

	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]

	st.write(predicted_value)