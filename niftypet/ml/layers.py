import logging
from collections import Iterable

from tensorflow import keras
from tensorflow import math as tfm

log = logging.getLogger(__name__)


def nrmse(y_true, y_pred):
    return tfm.sqrt(
        tfm.reduce_mean(tfm.squared_difference(y_true, y_pred)) / tfm.reduce_mean(y_true**2))


class Norm(keras.layers.Layer):
    """per-channel mean and std normalisation"""
    def __init__(self, mean=False, std=True, eps=1e-6, batch=False, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.batch = batch
        super(Norm, self).__init__(**kwargs)

    def get_config(self):
        config = super(Norm, self).get_config().copy()
        config.update(mean=self.mean)
        config.update(std=self.std)
        config.update(eps=self.eps)
        config.update(batch=self.batch)
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        input_shape = x.shape.as_list()
        axis = tuple(range(0 if self.batch else 1, len(input_shape) if input_shape else 4))
        if self.mean:
            x = x - tfm.reduce_mean(x, axis=axis, keepdims=True)
            if self.std:
                std = tfm.sqrt(tfm.reduce_mean(tfm.square(x), axis=axis, keepdims=True))
        elif self.std:
            std = tfm.reduce_std(x, axis=axis, keepdims=True)
        return x / ((self.eps + std) if self.eps else std) if self.std else x


class LocalityAdaptive(keras.layers.Layer):
    """
    Locality Adaptive[1]: per-ROI convolution.

    >>> x = keras.backend.variable(np.arange(3*64*32*2).reshape(3,64,32,2))
    >>> lALayer = LocalityAdaptive(roi_size=32)
    >>> out = lALayer(x)
    >>> lALayer.set_weights([np.array([[0.5, 0.5], [0.9, 0.1]])[:, None, :, None]])
    >>> out = lALayer(x)
    >>> assert out.shape[:-1] == x.shape[:-1]
    >>> assert keras.backend.all(sum(x[0, 1, 1] * [0.5, 0.5]) - out[0, 1, 1] < 1e-6)
    >>> assert keras.backend.all(sum(x[0, -1, -1] * [0.9, 0.1]) - out[0, -1, -1] < 1e-6)

    [1] Wang et al. 2019 TMI 38(6) 1328-1339
    "3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis"
    """
    def __init__(self, filters=1, roi_size=32, **kwargs):
        assert filters == 1, NotImplementedError
        self.filters = filters
        self.roi_size = roi_size
        super(LocalityAdaptive, self).__init__(**kwargs)
        # assert self.data_format and self.data_format != "channels_first"

    def get_config(self):
        config = super(LocalityAdaptive, self).get_config().copy()
        config.update(filters=self.filters)
        config.update(roi_size=self.roi_size)
        return config

    def build(self, input_shape):
        in_dims = input_shape[1:-1]
        roi = self.roi_size
        if not isinstance(roi, Iterable):
            roi = (roi,) * len(in_dims)

        in_ch = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=tuple((i+j-1) // j for i, j in zip(in_dims, roi)) + (in_ch, self.filters),
            initializer=keras.initializers.constant(1.0 / in_ch),
            trainable=True,
        )
        super(LocalityAdaptive, self).build(input_shape)

    @staticmethod
    def _call_helper(x, knl, roi, axis=0):
        """
        Recursive helper using concatenation.
        Required since tensors cannot be assinged to.

        x  : shape [N, x0, ..., xn-1, C]
        knl  : shape [s0, ..., sn-1, C, filters]: weights
          Note that there are `s0*...*sn` patches,
          each with its own kernel of shape [1, C, filters]
        roi  : shape [r0, ..., rn-1]: patch shape
        axis  : in range(n), which axis to expand/concatenate over
        """
        assert axis >= 0
        # s0, ..., sn-1, C, filters
        shp_knl = knl.shape.as_list()
        padding = (slice(0, None),) * axis
        log.debug("knl.shape:{}, axis:{}".format(shp_knl, axis))
        KB = keras.backend
        if axis == len(shp_knl) - 3:
            conv = {3: KB.conv1d, 4: KB.conv2d, 5: KB.conv3d}[len(shp_knl)]
            out = [
                conv(
                    x[(slice(0, None),) + padding + (slice(i * roi[axis], (i+1) * roi[axis]),)],
                    knl[padding + (slice(i, i + 1),)],
                ) for i in range(shp_knl[axis])]
        else:
            out = [
                LocalityAdaptive._call_helper(
                    x[(slice(0, None),) + padding + (slice(i * roi[axis], (i+1) * roi[axis]),)],
                    knl[padding + (slice(i, i + 1),)],
                    roi,
                    axis=axis + 1,
                ) for i in range(shp_knl[axis])]
        return KB.concatenate(out, axis=1 + axis)

    def call(self, x):
        in_dims = x.shape.as_list()[1:-1]
        roi = self.roi_size
        if not isinstance(roi, Iterable):
            roi = (roi,) * len(in_dims)
        return self._call_helper(x, self.kernel, roi)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
