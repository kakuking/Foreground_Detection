import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, ConvLSTM2D, BatchNormalization, Activation, Concatenate, Input, RepeatVector, MaxPooling2D, UpSampling2D, TimeDistributed
from keras.models import Model, load_model
from keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

import keras.backend as K


from PIL import Image
import matplotlib.pyplot as plt
import random

def iou_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
    return 1 - iou  # IoU loss is often used as 1 - IoU for minimization

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
    
    model.save("./Models/model_V2_2.keras")
    return model

def eval_model(model, X_test, Y_test):

    loss = model.evaluate(X_test, Y_test)
    print(f"Test Lost: {loss}")
    
    sample_index = random.randint(0, len(X_test))
    # sample_index = 5
    sample_ip = np.expand_dims(X_test[sample_index], axis=0)
    predicted_mask = model.predict(sample_ip)
    
    print(predicted_mask.shape)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(X_test[sample_index][4].squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(Y_test[sample_index][4].squeeze(), cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    # print(predicted_mask[0][0].shape)
    plt.imshow(predicted_mask[0][4].squeeze(), cmap='gray')
    
    plt.show()

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

X_data = np.load("X_data.npy")[470:570]
Y_data = np.load("Y_data.npy")[470:570]


# MAKE THE DATA INTO SEQUENCES
sequence_length = 5
num_samples = Y_data.shape[0]
num_seq = num_samples - sequence_length + 1

X_reshaped = np.zeros((num_seq, sequence_length, X_data.shape[1], X_data.shape[2], X_data.shape[3]))
Y_reshaped = np.zeros((num_seq, sequence_length, Y_data.shape[1], Y_data.shape[2], Y_data.shape[3]))

for i in range(num_seq):
    X_reshaped[i] = X_data[i:i+sequence_length]
    Y_reshaped[i] = Y_data[i+sequence_length-1]        

X_data = X_reshaped
Y_data = Y_reshaped

# print(X_data.shape)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=TEST_RATIO + VAL_RATIO, random_state=42)    # the Ultimate Question of Life, the Universe, and Everything
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO), random_state=42)

# print(X_train.shape)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# model = create_model(input_shape)
# model = train_model(model, X_train, Y_train, X_val, Y_val, 1, 8)

model = load_model("./Models/model_V2_2.keras")

eval_model(model, X_test, Y_test)
