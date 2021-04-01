import warnings

from pytest import mark

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tensorflow import keras # NOQA

from niftypet import ml


@mark.parametrize("input_channels", [1, 2, 5])
@mark.parametrize("ndim", [1, 2, 3])
@mark.parametrize("model", ml.MODELS)
def test_models(ndim, input_channels, model):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='tensorflow')
        net = model((128,) * ndim + (input_channels,))
        assert hasattr(net, "fit")
