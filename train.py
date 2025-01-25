import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input


data_path = './dataset/train_converted'

image_generator = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                     validation_split=0.2,
                                     rotation_range=30,
                                     width_shift_range=10,
                                     height_shift_range=10,
                                     shear_range=10,
                                     zoom_range=0.1,
                                     vertical_flip=False
                                     )

train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=data_path,
                                                 shuffle=True,
                                                 target_size=(150, 150),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=data_path,
                                                 shuffle=False,
                                                 target_size=(150, 150),
                                                 subset="validation",
                                                 class_mode='categorical')



# function to create the model
def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(11)(drop)

    model = keras.Model(inputs, outputs)

    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model        

#checkpointing to save the most accurate model during training. The best model will be automatically saved
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v3_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)                                        

# create parameters for the training model
learning_rate = 0.001
size = 100
droprate = 0.2

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_dataset, epochs=15, validation_data=validation_dataset, callbacks=[checkpoint])