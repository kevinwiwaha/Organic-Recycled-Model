from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

base_model = tf.keras.applications.MobileNet()
base_model_v2 = tf.keras.applications.MobileNetV2()
output_layer = base_model_v2.get_layer('out_relu')
base_model_v2.trainable = False


x = layers.Flatten()(output_layer.output)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# # Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)


# Flatten the output layer to 1 dimension
model = Model(base_model_v2.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# print(os.listdir('./dataset'))

train_dir = './dataset/TRAIN'
validation_dir = './dataset/TEST'
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(240, 240))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(240, 240))


history = model.fit(
    train_generator,

    steps_per_epoch=10,
    epochs=5,

    verbose=1)
