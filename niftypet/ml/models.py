"""
PET-MR denoising CNNs.
"""
import functools
import logging

import numpy as np
from tensorflow import keras

from .layers import Norm as NormLayer
from .layers import nrmse

log = logging.getLogger(__name__)
L = keras.layers
CONV_ND = {2: L.Conv1D, 3: L.Conv2D, 4: L.Conv3D}
MAXPOOL_ND = {2: L.MaxPool1D, 3: L.MaxPool2D, 4: L.MaxPool3D}
UPSAMPLE_ND = {
    2: L.UpSampling1D, 3: functools.partial(L.UpSampling2D, interpolation="bilinear"),
    4: L.UpSampling3D}
AVGPOOL_ND = {2: L.AvgPool1D, 3: L.AvgPool2D, 4: L.AvgPool3D}


def dcl2021(input_shape, n_filters=None, filter_sizes=None, activations=None, prenorm=None, eps=0,
            lr=1e-3, dtype="float32"):
    """
    Micro-net implementation based on:
    C. O. da Costa-Luis and A. J. Reader 2021 IEEE Trans. Radiat. Plasma Med. Sci. 5(2) 202-212
    "Micro-Networks for Robust MR-Guided Low Count PET Imaging"

    Args:
      input_shape (tuple): (num_slices, [[Z,] Y,] X, num_channels),
        where channels has PET first.
      n_filters: default [32, 32, 1]
      filter_sizes: default [5, 3, 1]
      activations: default ['sigmoid', ..., 'sigmoid', 'elu']
      prenorm: default is `activation[-1] != 'sigmoid'`
      eps: used in prenorm (`NormLayer`)
    """
    Conv = CONV_ND[len(input_shape)]
    n_filters = n_filters or [32, 32, 1]
    filter_sizes = filter_sizes or [5, 3, 1]
    activations = activations or ["sigmoid"] * (len(n_filters) - 1) + ["elu"]

    x = inputs = L.Input(input_shape, dtype=dtype)

    largs = {
        'kernel_initializer': "he_normal", 'bias_initializer': "he_normal", 'padding': "same",
        'strides': 1}
    Norm = functools.partial(NormLayer, eps=eps, std=True, batch=True)

    inputs = x = L.Input(input_shape, dtype=dtype)
    if prenorm is None:
        prenorm = activations[-1] != 'sigmoid'
    if prenorm:
        x = L.concatenate([Norm(mean=False)(x[..., :1]), Norm(mean=True)(x[..., 1:])]) # PET, MR

    for filters, kernel_size, activation in zip(n_filters, filter_sizes, activations):
        x = Conv(filters, kernel_size, activation=activation, **largs)(x)
    # x = L.Multiply()((x, std))  # un-norm

    model = keras.Model(inputs=inputs, outputs=x)
    if lr:
        opt = keras.optimizers.Adam(lr)
        model.compile(opt, metrics=[nrmse], loss=nrmse)
        model.summary(print_fn=log.debug)
    return model


def xu2020(input_shape, residual_input_channel=0, lr=1e-3, dtype="float32"):
    """
    Residual U-net implementation based on:
    J. Xu, et al. 2020 Medical Imaging: Image Process. p. 60
    "Ultra-low-dose 18F-FDG brain PET/MR denoising using deep learning
    and multi-contrast information"

    Args:
      input_shape (tuple): (num_slices, [[Z,] Y,] X, num_channels),
        where channels has PET first.
      residual_input_channel(int): input channel index to use for residual addition
        [default: 0] for PET.
    """
    Conv = CONV_ND[len(input_shape)]
    AvgPool = AVGPOOL_ND[len(input_shape)]
    Upsample = UPSAMPLE_ND[len(input_shape)]

    x = inputs = L.Input(input_shape, dtype=dtype)

    def block(x, filters):
        x = Conv(filters, 3, padding="same", use_bias=False, dtype=dtype)(x)
        x = L.BatchNormalization(dtype=dtype)(x)
        x = L.LeakyReLU(dtype=dtype)(x) # WARN: alpha value?
        return x

    # U-net
    filters = [32, 64, 128, 256]
    # # encode
    convs = []
    for i in filters[:-1]:
        x = residual = block(x, filters=i)
        x = block(x, filters=i)
        x = L.Add()([x, residual])
        convs.append(x)
        x = AvgPool(dtype=dtype, padding="same")(x)
    x = residual = block(x, filters=filters[-1])
    x = block(x, filters=filters[-1])
    x = L.Add()([x, residual])
    # # decode
    for i in filters[:-1][::-1]:
        x = Upsample(dtype=dtype)(x)
        x = L.Concatenate()([x, convs.pop()])
        x = residual = block(x, filters=i)
        x = block(x, filters=i)
        x = L.Add()([x, residual])
    x = Conv(1, 1, padding="same", dtype=dtype, name="residual")(x)
    x = L.Add(name="generated")([
        x, inputs[..., residual_input_channel:residual_input_channel + 1]])

    model = keras.Model(inputs=inputs, outputs=x)
    if lr:
        opt = keras.optimizers.RMSprop(lr)
        model.compile(opt, metrics=[nrmse], loss="mae")
        model.summary(print_fn=log.debug)
    return model


def chen2019(input_shape, residual_input_channel=0, lr=2e-4, dtype="float32"):
    """
    Residual U-net implementation based on:
    K. T. Chen et al. 2019 Radiol. 290(3) 649-656
    "Ultra-Low-Dose 18F-Florbetaben Amyloid PET Imaging Using Deep Learning
    with Multi-Contrast MRI Inputs"

    >>> model = network(input_data.shape[1:])
    >>> model.fit(input_data, output_date, epochs=100, batch_size=input_data.shape[0] // 4, ...)

    Args:
      input_shape (tuple): (num_slices, [[Z,] Y,] X, num_channels),
        where channels has PET first.
      residual_input_channel(int): input channel index to use for residual addition
        [default: 0] for PET.
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
        x = Upsample(dtype=dtype)(x)
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




def grid(input_shape, n_filters=None, filter_sizes=None, activations=None, concat=None,
         strides=None, eps=0, l1reg=0, lr=1e-3, salt=None, dtype="float32"):
    """
    Generic network with customisable:
    - filters per layer
    - filter sizes
    - skip (concat) connections
    - downsampling (encoding)
    - upsampling (decoding)

    Also inspired from other works:
    - dcl2021: sigmoid activations except last layer ELU
    - dcl2021: He normal weights initialisation
    - dcl2021: Initial normalisation (mean & std for MR; std for PET)
    - dcl2021: NRMSE loss
    - kaplan2018: l1 regularisation
    - downsampling: strided convolution
    - upsampling: bilinear interpolation followed by stride-1 convolution

    Args (in order of precedence):
      strides: e.g. [2, 0.5] for a two-layer encoder-decoder network.
        N.B. factional stride is actually implemented as
        bilinear upsampling & stride-1 convolution.
      concat: e.g. [(0, 2), ...] would concatenate (skip)
        the input layer with the second convolutional layer's output
        (forming a new "second" layer output)
      n_filters: default [32, 32, 1]
      filter_sizes: default [3, ..., 3, 1]
      activations: default ['sigmoid', ..., 'sigmoid', 'elu']

    e.g. U-net:
      strides=[1, 2, 2, 2, 0.5, 0.5, 0.5, 1, 1]
      concat=[(0, 8), (1, 7), (2, 6), (3, 5)]
      n_filters=[32, 64, 128, 256, 128, 64, 32, 1, 1]
    """
    Conv = CONV_ND[len(input_shape)]
    Upsample = UPSAMPLE_ND[len(input_shape)]

    n_filters = n_filters or [32, 32, 1]
    filter_sizes = filter_sizes or [3] * (len(n_filters) - 1) + [1]
    if not activations:
        # number of "end" layers, i.e. same number of filters
        # e.g. [32, 32, 1, 1] -> 2 ends
        ends = [i == n_filters[-1] for i in n_filters[::-1]]
        ends = ends.index(False) if False in ends else len(n_filters)
        activations = ["sigmoid"] * (len(n_filters) - ends) + ["elu"] * ends
    concat = concat or []
    strides = strides or [1] * len(n_filters)

    Norm = functools.partial(NormLayer, eps=eps, std=True, batch=True)
    largs = {'kernel_initializer': 'he_normal', 'bias_initializer': 'he_normal', 'padding': 'same'}
    if l1reg:
        largs.update(
            kernel_regularizer=keras.regularizers.l1(l1reg),
            bias_regularizer=keras.regularizers.l1(l1reg),
        )

    x = inputs = L.Input(input_shape, dtype=dtype)
    x = L.concatenate([Norm(mean=False)(x[..., :1]), Norm(mean=True)(x[..., 1:])]) # PET, MR

    layers = [x]
    for i, (n, s, a, d) in enumerate(zip(n_filters, filter_sizes, activations, strides), 1):
        log.debug(i, (n, s, a, d))
        if d < 1:
            log.debug("upsample")
            x = Upsample(int(np.round(1 / d)))(x)
        c = [layers[cin] for cin, cout in concat if cout == i]
        if c:
            log.debug("concat:", c)
            if i == len(n_filters) and c == [layers[0]]:
                # last layer & input so extract PET input
                x = L.concatenate(c + [x[..., 1:]])
            else:
                x = L.concatenate(c + [x])
        log.debug("conv")
        x = Conv(filters=n, kernel_size=s, activation=a, strides=d if d > 1 else 1, **largs)(x)
        layers.append(x)

        # x = L.Multiply()((x, std))  # un-norm

    model = keras.Model(inputs=inputs, outputs=x)
    if lr:
        opt = keras.optimizers.Adam(lr)
        model.compile(opt, metrics=[nrmse], loss=nrmse)
        model.summary(print_fn=log.debug)
    return model


MODELS = [dcl2021, xu2020, chen2019, grid]
