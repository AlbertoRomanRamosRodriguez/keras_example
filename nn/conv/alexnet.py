from keras.models import Sequential

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

from keras.regularizers import L2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width: int, height: int, depth:int, classes:int, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() ==  "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(
            96,
            (11,11),
            strides=(4,4),
            input_shape=inputShape,
            padding="same",
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2)
        ))
        model.add(Dropout(0.25))
        model.add(Conv2D(
            256,
            (5,5),
            padding="same",
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2)
        ))
        model.add(Dropout(0.25))
        model.add(Conv2D(
            384,
            (3,3),
            padding="same",
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(
            384,
            (3,3),
            padding="same",
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(
            256,
            (5,5),
            padding="same",
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2)
        ))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(
            4096,
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(
            classes,
            kernel_regularizer=L2(reg)
        ))
        model.add(Activation("softmax"))

        return model