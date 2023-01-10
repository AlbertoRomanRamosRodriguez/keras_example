from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class ShallowNet:
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height,width,depth)
        filterShape = (3,3)

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
        
        model.add(Conv2D(32,filterShape, padding="same", input_shape=inputShape))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model