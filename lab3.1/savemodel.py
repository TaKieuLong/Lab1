import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights="imagenet")

model.save("model.h5")