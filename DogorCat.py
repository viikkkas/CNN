import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#preprocessing the training set (avoid overfitting)
train_datagen = ImageDataGenerator(
        rescale=1./255,   #feature scaling
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#initializing the cnn
cnn = tf.keras.models.Sequential()

#convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#adding second cnn layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the cnn on training set and evaluating it on test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#making prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/Prediction/dog.4020.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog';
else:
    prediction = 'cat';
print(prediction)
