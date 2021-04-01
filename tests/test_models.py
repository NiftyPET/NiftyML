import warnings

from pytest import mark

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tensorflow import keras # NOQA

from niftypet import ml


@mark.parametrize("ndim,input_channels", [(2, 1), (3, 1), (2, 5), (3, 5)])
def test_models(ndim, input_channels):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='tensorflow')
        _ = ml.dcl2021((128,) * ndim + (input_channels,))
