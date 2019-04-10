from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.models import Model

def Discriminator(input_shape, n_filters=64, activation_fn="relu"):
    inputs = Input(shape=input_shape)
    x = Conv2D(n_filters, (4, 4), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = Conv2D(n_filters*2, (4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = Conv2D(n_filters*4, (4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = Conv2D(n_filters*8, (4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)
    x = Conv2D(1, (4, 4), strides=(2, 2), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model