import warnings

from pytest import mark, skip

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tensorflow import keras # NOQA

from niftypet import ml


@mark.parametrize("input_channels", [1, 2])
@mark.parametrize("ndim", [1, 2, 3])
@mark.parametrize("model", ml.MODELS)
def test_models(ndim, input_channels, model):
    try:
        if issubclass(model, ml.models.GAN) and ndim == 1:
            skip("GAN must have ndim>1")
    except TypeError:
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='tensorflow')
        net = model((128,) * ndim + (input_channels,))
        assert hasattr(net, "fit") or isinstance(net, ml.models.GAN)


@mark.parametrize("input_channels", [1, 2])
@mark.parametrize("ndim", [1, 2, 3])
def test_grid_Unet(ndim, input_channels):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='tensorflow')
        net = ml.grid(
            (128,) * ndim + (input_channels,),
            strides=[1, 2, 2, 2, 0.5, 0.5, 0.5, 1, 1],
            concat=[(0, 8), (1, 7), (2, 6), (3, 5)],
            n_filters=[32, 64, 128, 256, 128, 64, 32, 1, 1])  # yapf: disable
        assert hasattr(net, "fit")
