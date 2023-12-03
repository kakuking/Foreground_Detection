import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, ConvLSTM2D, BatchNormalization, Activation, Concatenate, Input, RepeatVector, MaxPooling2D, UpSampling2D, TimeDistributed
from keras.models import Model, load_model
from keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
import keras.backend as K

from skimage import exposure
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import random

def iou_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
    return 1 - iou  # IoU loss is often used as 1 - IoU for minimization

'''def create_model(ip_shape):
    # inputs = Input(shape=ip_shape)

    model = models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=ip_shape, padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=ip_shape, padding='same'))
    model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), input_shape=ip_shape, padding='same'))
    model.add(layers.BatchNormalization())

    # Add more ConvLSTM layers as needed

    # model.add(layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model'''

def create_model(ip_shape):
    inputs = Input(shape=ip_shape)

    # ENCODER PART
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

    # DECODER PART
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = TimeDistributed(BatchNormalization())(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, num_epochs, batch_size):
    model.fit(
        X_train,
        Y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val)
    )
    
    model.save("./Models/model_Vtemp_V2.keras")
    return model

def eval_model(model, X_test, Y_test, idx):

    loss = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Lost: {loss}")
    
    sample_index = random.randint(0, len(X_test)) if idx == -1 else idx
    # sample_index = 5
    sample_ip = np.expand_dims(X_test[sample_index], axis=0)
    predicted_mask = model.predict(sample_ip, verbose=0)
    
    op_img = predicted_mask.squeeze()
    op_eq = exposure.equalize_hist(op_img)
    
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title('Input Image')
    plt.imshow(X_test[sample_index].squeeze(), cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(Y_test[sample_index].squeeze(), cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title('Predicted Mask Contrasted')
    plt.imshow(op_eq, cmap='gray')
    
    plt.savefig(f"./SS/{idx}.jpg")
    if(idx == -1):
        plt.show()

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

X_data = np.load("X_data.npy")[470:]
Y_data = np.load("Y_data.npy")[470:]

print(X_data.shape)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=TEST_RATIO + VAL_RATIO, random_state=42)    # the Ultimate Question of Life, the Universe, and Everything
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO), random_state=42)

# print(X_train.shape)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# model = create_model(input_shape)
# model = train_model(model, X_train, Y_train, X_val, Y_val, 40, 8)

model = load_model("./Models/model_Vtemp_V2.keras")

for i in tqdm(range(0, 1000, 200)):
    eval_model(model, X_data, Y_data, i)
