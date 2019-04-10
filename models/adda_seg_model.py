from keras.models import Model
from keras.layers import Input

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


# def GAN_model(latent_dim, _generator, _discriminator):
#     set_trainability(_discriminator, False)
#     gan_input = Input(shape=(latent_dim,))
#     x = _generator(gan_input)
#     gan_output = _discriminator(x)
#     gan = Model(gan_input, gan_output)
#     gan.compile(optimizer=optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8), loss="binary_crossentropy")
#     return gan


def ADDA_Seg_Model(g_input_shape, G, D):
    set_trainability(D, False)
    g_input = Input(shape=g_input_shape)
    seg_output = G(g_input)
    d_output = D(seg_output)

    model = Model(g_input, d_output)
    return model