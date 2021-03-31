"""
PET-MR denoising CNNs.
"""
import functools
import logging

from tensorflow import keras

from .layers import Norm as NormLayer
from .layers import nrmse

__author__ = "Casper da Costa-Luis <imaging@cdcl.ml>"
log = logging.getLogger(__name__)
L = keras.layers
CONV_ND = {2: L.Conv1D, 3: L.Conv2D, 4: L.Conv3D}
MAXPOOL_ND = {2: L.MaxPool1D, 3: L.MaxPool2D, 4: L.MaxPool3D}
UPSAMPLE_ND = {2: L.UpSampling1D, 3: L.UpSampling2D, 4: L.UpSampling3D}


def dcl2020(input_shape, n_filters=None, filter_sizes=None, activations=None, prenorm=None, eps=0,
            lr=1e-3, dtype="float32"):
    """
    Micro-net implementation based on:
    C. O. da Costa-Luis and A. J. Reader 2021 IEEE Trans. Radiat. Plasma Med. Sci. 5(2) 202-212
    "Micro-Networks for Robust MR-Guided Low Count PET Imaging"

    Args:
      n_filters: default [32, 32, 1]
      filter_sizes: default [5, 3, 1]
      activations: default ['sigmoid', ..., 'sigmoid', 'elu']
      prenorm: default is `activation[-1] != 'sigmoid'`
      wd: weigth for discriminator loss term
    """
    n_filters = n_filters or [32, 32, 1]
    filter_sizes = filter_sizes or [5, 3, 1]
    activations = activations or ["sigmoid"] * (len(n_filters) - 1) + ["elu"]

    Conv = CONV_ND[len(input_shape)]

    x = inputs = L.Input(input_shape, dtype=dtype)

    largs = {
        'kernel_initializer': "he_normal", 'bias_initializer': "he_normal", 'padding': "same",
        'strides': 1}
    Norm = functools.partial(NormLayer, eps=eps, std=True, batch=True)

    inputs = x = L.Input(input_shape, dtype=dtype)
    if prenorm is None:
        prenorm = activations[-1] != 'sigmoid'
    if prenorm:
        x = L.concatenate([Norm(mean=True)(x[..., :1]), Norm(mean=False)(x[..., 1:])]) # MR  # PET

    for filters, kernel_size, activation in zip(n_filters, filter_sizes, activations):
        x = Conv(filters, kernel_size, activation=activation, **largs)(x)
    # x = L.Multiply()((x, std))  # un-norm

    model = keras.Model(inputs=inputs, outputs=x)
    if lr:
        opt = keras.optimizers.Adam(lr)
        model.compile(opt, metrics=[nrmse], loss=nrmse)
        model.summary(print_fn=log.debug)
    return model


def chen2019(input_shape, residual_input_channel=1, lr=2e-4, dtype="float32"):
    """
    Residual U-net implementation based on:
    K. T. Chen et al. 2019 Radiol. 290(3) 649-656
    "Ultra-Low-Dose 18F-Florbetaben Amyloid PET Imaging Using Deep Learning
    with Multi-Contrast MRI Inputs"

    >>> model = network(input_data.shape[1:])
    >>> model.fit(input_data, output_date, epochs=100, batch_size=input_data.shape[0] // 4, ...)

    Args:
      input_shape (tuple): (num_slices, slice_height, slice_width, num_channels)
      residual_input_channel  : input channel index to use for residual addition
    """
    Conv = CONV_ND[len(input_shape)]
    MaxPool = MAXPOOL_ND[len(input_shape)]
    Upsample = UPSAMPLE_ND[len(input_shape)]

    x = inputs = L.Input(input_shape, dtype=dtype)

    def block(x, filters):
        x = Conv(filters, 3, padding="same", use_bias=False, dtype=dtype)(x)
        x = L.BatchNormalization(dtype=dtype)(x)
        x = L.ReLU(dtype=dtype)(x)
        return x

    # U-net
    filters = [16, 32, 64, 128]
    # # encode
    convs = []
    for i in filters[:-1]:
        x = block(x, i)
        x = block(x, i)
        convs.append(x)
        x = MaxPool(dtype=dtype, padding="same")(x)
    x = block(x, filters[-1])
    x = block(x, filters[-1])
    # # decode
    for i in filters[:-1][::-1]:
        x = Upsample(interpolation="bilinear", dtype=dtype)(x)
        x = L.Concatenate()([x, convs.pop()])
        x = block(x, i)
        x = block(x, i)
    x = Conv(1, 1, padding="same", dtype=dtype, name="residual")(x)
    x = L.Add(name="generated")([
        inputs[..., residual_input_channel:residual_input_channel + 1], x])

    model = keras.Model(inputs=inputs, outputs=x)
    if lr:
        opt = keras.optimizers.Adam(lr)
        model.compile(opt, metrics=[nrmse], loss="mae")
        model.summary(print_fn=log.debug)
    return model
