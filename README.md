# Road Sign Classification using Convolutional Neural Networks

## Overview
This project aims to classify road signs using deep learning, specifically Convolutional Neural Networks (CNNs). The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains images of road signs from 43 different classes. The goal is to create an accurate model that can classify road sign images into one of these 43 categories.

## Dataset
The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which includes images of road signs belonging to the following categories:

* 43 classes representing different types of road signs (e.g., stop signs, yield signs, speed limits, etc.).
* Each image is resized to fit a standard input size (in this case, 30x30 pixels).

For more information about the dataset, visit: [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

## Model Architecture
The model is built using TensorFlow and Keras. Below is a summary of the architecture:

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Conv2D(16, (2, 2), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Training and Validation Results

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - accuracy: 0.1127 - loss: 6.5972     
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - accuracy: 0.5171 - loss: 1.6939  
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.7092 - loss: 0.9712  
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - accuracy: 0.8139 - loss: 0.6221  
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.8626 - loss: 0.4559 
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.8826 - loss: 0.3844 
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9103 - loss: 0.3163 
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9177 - loss: 0.2767 
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9252 - loss: 0.2638 
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9295 - loss: 0.2459 
333/333 - 1s - 4ms/step - accuracy: 0.9734 - loss: 0.0994
```
