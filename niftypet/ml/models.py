"""
PET-MR denoising CNNs.
"""
import functools
import logging
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import math as tfm

from .layers import LocalityAdaptive
from .layers import Norm as NormLayer
from .layers import PSFInit, blur, gradAndSq2D, gradAndSq3D, nrmse

log = logging.getLogger(__name__)
L = keras.layers
CONV_ND = {2: L.Conv1D, 3: L.Conv2D, 4: L.Conv3D}
CONVT_ND = {3: L.Conv2DTranspose, 4: L.Conv3DTranspose}
MAXPOOL_ND = {2: L.MaxPool1D, 3: L.MaxPool2D, 4: L.MaxPool3D}
UPSAMPLE_ND = {
    2: L.UpSampling1D, 3: functools.partial(L.UpSampling2D, interpolation="bilinear"),
    4: L.UpSampling3D}
AVGPOOL_ND = {2: L.AvgPool1D, 3: L.AvgPool2D, 4: L.AvgPool3D}
GRADSQ_ND = {3: gradAndSq2D, 4: gradAndSq3D}


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


class GAN(object):
    @staticmethod
    def lossGen(y_true, y_pred):
        """Generator loss (pre-discriminator)"""
        return keras.losses.mean_squared_error(y_true, y_pred)

    def lossDsc(self, y_true, y_pred, loss=keras.losses.binary_crossentropy):
        """
        Discriminator loss

        loss(self.modelDsc(y_true), ones) + loss(self.modelDsc(y_pred), zeros)
        """
        ones = self.modelDsc(y_true)
        zeros = self.modelDsc(y_pred)
        return tfm.reduce_mean(loss(tf.ones_like(ones), ones) + loss(tf.zeros_like(zeros), zeros))

    def _generator(self, input_shape):
        """Returns a keras.Model"""
        raise NotImplementedError

    def _discriminator(self, input_shape):
        """Returns a keras.Model"""
        raise NotImplementedError

    @contextmanager
    def contextDsc(self):
        """Usage:
        ```
        with self.contextDsc() as modelDsc:
            modelDsc.train_on_batch(x, y_class)
        modelGAN.train_on_batch(x, y)
        ```
        """
        self.modelGAN.trainable = False
        self.modelGen.trainable = False
        self.modelDsc.trainable = True
        yield self.modelDsc # WARN: should be modelAdv?
        self.modelDsc.trainable = False
        self.modelGen.trainable = True
        self.modelGAN.trainable = True


class GANConstant(GAN):
    """GAN with custom list of (non-traininable) constants"""
    def __init__(self, *args, **kwargs):
        super(GANConstant, self).__init__(*args, **kwargs)
        self.constants = []

    def ensureConstants(self):
        for i in self.constants:
            i.trainable = False

    @contextmanager
    def contextDsc(self):
        with super(GANConstant, self).contextDsc() as modelDsc:
            self.ensureConstants()
            yield modelDsc
        self.ensureConstants()


class Wang2019(GAN):
    """
    Locality Adaptive U-net GAN implementation based on:
    Wang et al. 2019 TMI 38(6) 1328-1339
    "3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis"
    """
    @staticmethod
    def preProc(inputs, sigma=2, eps=0, channels=None):
        """channels (list): (default: [0]) for PET"""
        inputs = inputs.copy()
        for ch in channels or [0]:
            inputs[..., ch] = (blur(inputs[..., ch], sigma=(0,) + (sigma,) *
                                    (inputs.ndim - 2), mode="constant") + eps)
        return inputs

    def __init__(
            self,
            input_shape,
            lr=2e-4,                  # WARN: lr decay?
            w1=200,
            residual_input_channel=0,
            dtype="float32"):
        """
        Args:
          lr: learning rate
          l1reg: traininable parameter L1 regularisation weight
          w1: weight of generator loss
          residual_input_channel: for PET
        """
        self.modelGen = self._generator(input_shape, lr=0, dtype=dtype)
        self.modelGen.trainable = False

        self.modelDsc = self._discriminator(self.modelGen.output_shape[1:-1] + (2,), lr=lr,
                                            dtype=dtype)
        self.modelDsc.trainable = False

        # GAN
        x = inputs = L.Input(input_shape, dtype=dtype)
        inputs_low = inputs[..., residual_input_channel:residual_input_channel + 1]
        x = generated = self.modelGen(x)
        x = L.concatenate([inputs_low, generated])
        x = discriminated = self.modelDsc(x)

        model = keras.Model(inputs=inputs, outputs=[generated, discriminated], name="GAN")

        self.modelGen.trainable = True
        opt = keras.optimizers.Adam(lr)
        model.compile(opt, metrics=[[nrmse], ["accuracy"]], loss=["mae", "binary_crossentropy"],
                      loss_weights=[w1, -1])
        model.summary(print_fn=log.debug)
        self.modelGAN = model
        self.modelGAN.trainable = False

    def _get_block(self, input_shape, dtype="float32"):
        Conv = CONV_ND[len(input_shape)]
        ConvTranspose = CONVT_ND[len(input_shape)]

        def block(x, filters, transpose=False, padding="same", **kwargs):
            x = (ConvTranspose if transpose else Conv)(filters, 4, strides=2, padding=padding,
                                                       use_bias=False, dtype=dtype)(x)
            x = L.BatchNormalization(dtype=dtype)(x)
            x = L.LeakyReLU(0.2, dtype=dtype, **kwargs)(x)
            return x

        return block

    def _generator(self, input_shape, lr=2e-4, dtype="float32", dropout=0):
        """
        @return `Model` mapping `input_shape` => `input_shape[:-1] + (1,)`
        """
        block = self._get_block(input_shape, dtype=dtype)

        x = inputs = L.Input(input_shape, dtype=dtype)

        x = LocalityAdaptive(roi_size=8)(x) # WARN: dtype

        encLayer = functools.partial(block, transpose=False)

        def decLayer(x, filters, concat=None, name=None):
            kwargs = {}
            if concat is None and name is not None:
                kwargs['name'] = name
            x = block(x, filters, transpose=True, **kwargs)
            if concat is not None:
                if name is not None:
                    kwargs['name'] = name
                x = L.Concatenate(**kwargs)([x, concat])
            return x

        # U-net
        filters = [64, 128, 256, 512, 512, 512]
        # # encode
        convs = []
        for i in filters:
            x = encLayer(x, filters=i)
            convs.append(x)
        convs.pop()
        # # decode
        for i in filters[:-1][::-1]:
            x = decLayer(x, filters=i, concat=convs.pop())
        x = decLayer(x, filters=1, name="generated")

        model = keras.Model(inputs=inputs, outputs=x, name="Gen")
        if lr:
            opt = keras.optimizers.Adam(lr)
            model.compile(opt, metrics=[nrmse], loss="mae")
            model.summary(print_fn=log.debug)
        return model

    def _discriminator(self, input_shape, lr=2e-4, dtype="float32", dropout=0):
        """
        @return `Model` mapping `input_shape` => `(1,)`
        """
        Conv = CONV_ND[len(input_shape)]
        block = self._get_block(input_shape, dtype=dtype)

        x = L.Input(input_shape, dtype=dtype)
        inputs = x

        filters = [64, 128, 256, 512]
        for i in filters:
            x = block(x, filters=i)
        # fully connected isn't the same as dense: need output shape 1, not H x W( x D) x 1
        x = Conv(1, x.shape[1:-1].as_list(), padding="valid", activation="sigmoid")(x)
        x = L.Reshape((1,), name="discriminated")(x)

        model = keras.Model(inputs=inputs, outputs=x, name="Dsc")
        if lr:
            opt = keras.optimizers.Adam(lr)
            model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
            model.summary(print_fn=log.debug)
        return model


class Kaplan2018(GANConstant):
    """
    U-net GAN with TV & gradient loss implementation based on:
    Kaplan & Zhu 2018 J. Digit. Imaging 3
    "Full-Dose PET Image Estimation from Low-Dose PET Image Using Deep Learning: a Pilot Study"
    """
    def _get_lossGen(self, input_shape, w1, w2, w3):
        gradAndSq = GRADSQ_ND[len(input_shape)]

        def genLoss(y_true, y_pred):
            gradT, _ = gradAndSq(y_true)
            gradP, gradSqP = gradAndSq(y_pred)
            return (w1 * tfm.reduce_mean(tfm.squared_difference(y_true, y_pred)) +
                    w2 * tfm.reduce_mean(gradSqP) +
                    w3 * tfm.reduce_mean(tfm.squared_difference(gradT, gradP)))

        return genLoss

    def _get_block(self, input_shape, dtype="float32"):
        Conv = CONV_ND[len(input_shape)]
        ConvTranspose = CONVT_ND[len(input_shape)]

        def block(x, filters, kernel_size=3, transpose=False, padding="same", activation="elu",
                  strides=2, **largs):
            x = (ConvTranspose if transpose else Conv)(filters, kernel_size, strides=strides,
                                                       activation=activation, padding=padding,
                                                       dtype=dtype, **largs)(x)
            return x

        return block

    def __init__(self, input_shape, sigma=1.5, w1=1, w2=-5e-5, w3=0.075, w4=0.1, n_filters=None,
                 activations=None, residual_input_channel=0, lr=1e-3, l1reg=1e-4, dtype="float32"):
        """
        Args:
          lr: learning rate
          l1reg: traininable parameter L1 regularisation weight
          w1, w2, w3: generator loss weights
          w4: discriminator loss weight
          n_filters: for generator, discriminator layers (default: [128, 64])
          activations: for generator, discriminator layers (default: ["elu", "tanh"])
        """
        super(Kaplan2018, self).__init__()
        if n_filters is None:
            n_filters = [128, 64]
        if activations is None:
            activations = ["elu", "tanh"]
        largs = {
            'dtype': dtype, 'lr': lr,
            'kernel_regularizer': keras.regularizers.l1(l1reg) if l1reg else None,
            'bias_regularizer': keras.regularizers.l1(l1reg) if l1reg else None}

        self.modelGen = self._generator(input_shape, sigma=sigma, filters=n_filters[0], strides=2,
                                        activation=activations[0], padding="same", w1=w1, w2=w2,
                                        w3=w3, residual_input_channel=residual_input_channel,
                                        **largs)
        self.modelGen.trainable = False

        self.modelDsc = self._discriminator(self.modelGen.output_shape[1:], filters=n_filters[1],
                                            strides=1, activation=activations[1], **largs)
        self.modelDsc.trainable = False

        # GAN
        x = inputs = L.Input(input_shape)
        x = generated = self.modelGen(x)
        # inputs_low = inputs[..., -1:]
        # x = L.concatenate([inputs_low, generated])
        x = discriminated = self.modelDsc(x)

        model = keras.Model(inputs=inputs, outputs=[generated, discriminated], name="GAN")

        # WARN: adam_lr(init=lr, epoch=100) / 10
        opt = keras.optimizers.Adam(lr / 10)
        self.modelGen.trainable = True
        self.modelDsc.trainable = False
        self.ensureConstants()
        model.compile(
            opt,
            metrics=[[nrmse], ["accuracy"]],
            loss=[self._get_lossGen(input_shape, w1, w2, w3), "binary_crossentropy"],
            loss_weights=[1, -w4],
        )
        model.summary(print_fn=log.debug)
        self.modelGAN = model
        self.modelGAN.trainable = False

    @staticmethod
    def preProc(inputs, sigma=1.5, eps=0):
        return blur(inputs, sigma=(0,) + (sigma,) * (inputs.ndim - 2) +
                    (0,), mode="constant") + eps

    def _generator(self, input_shape, sigma=1.5, filters=128, w1=1, w2=-5e-5, w3=0.075, lr=1e-3,
                   residual_input_channel=0, dtype="float32", pre_filter=True, **largs):
        """
        Args:
          pre_filter: if [default: True], no need to manually call `preProc()`
          **largs:  anything supported by tf.keras.layers.Conv*
        """
        Conv = CONV_ND[len(input_shape)]
        block = self._get_block(input_shape, dtype=dtype)
        block = functools.partial(block, **largs)

        x = inputs = L.Input(input_shape, dtype=dtype)
        if pre_filter:
            preProc = Conv(
                input_shape[-1],
                int(np.ceil(sigma * 4)) * 2 + 1,
                padding="same",
                use_bias=False,
                kernel_initializer=PSFInit(sigma=(0,) * (input_shape[-1] - 1) + (sigma,)),
                dtype=dtype,
            )
            self.constants.append(preProc)
            x = filtered_inputs = preProc(x)
        else:
            filtered_inputs = x

        # U-net
        # # encode
        convs = []
        for _ in range(4):
            x = block(x, filters, transpose=False)
            convs.append(x)
        convs.pop()
        # # decode
        for _ in range(len(convs)):
            x = block(x, filters, transpose=True)
            x = L.Add()([x, convs.pop()])
        # # residual
        x = block(x, 1, transpose=True)
        x = L.Add(name="generated")([
            filtered_inputs[..., residual_input_channel:residual_input_channel + 1], x])

        model = keras.Model(inputs=inputs, outputs=x, name="Gen")
        if lr:
            opt = keras.optimizers.Adam(lr)
            self.ensureConstants()
            model.compile(opt, metrics=[nrmse], loss=self._get_lossGen(input_shape, w1, w2, w3))
            model.summary(print_fn=log.debug)
        return model

    def _discriminator(self, input_shape, filters=64, lr=1e-3, dtype="float32", **largs):
        block = self._get_block(input_shape, dtype=dtype)
        block = functools.partial(block, **largs)

        x = inputs = L.Input(input_shape, dtype=dtype)
        x = block(x, filters, padding="same")
        # fully connected isn't the same as dense: need output shape 1, not H x W( x D) x 1
        # x = block(x, 1, kernel_size=x.shape[1:-1].as_list(), padding="valid")
        x = block(x, 1, kernel_size=16, padding="valid")
        # per-patch prediction
        x = L.Activation("sigmoid")(x)
        # average patches
        x = L.Lambda(lambda i: tfm.reduce_mean(i, axis=range(1, len(input_shape))),
                     name="discriminated")(x)
        # x = L.Reshape((1,))(x)

        model = keras.Model(inputs=inputs, outputs=x, name="Dsc")
        if lr:
            opt = keras.optimizers.Adam(lr)
            self.ensureConstants()
            model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
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
    - Kaplan2018: l1 regularisation
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


MODELS = [dcl2021, xu2020, chen2019, Wang2019, Kaplan2018, grid]
