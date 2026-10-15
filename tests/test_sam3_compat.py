from types import SimpleNamespace

from spatialsam.models.sam3_compat import drop_unsupported_kwargs, patch_init_state_kwargs


def init_state(resource_path, offload_video_to_cpu=False):
    return (resource_path, offload_video_to_cpu)


def test_drop_unsupported_kwargs_filters_unknown():
    wrapped = drop_unsupported_kwargs(init_state)
    assert wrapped(resource_path="v", offload_video_to_cpu=True, offload_state_to_cpu=True) == ("v", True)


def test_drop_unsupported_kwargs_keeps_var_keyword_functions():
    def accepts_all(**kwargs):
        return kwargs

    assert drop_unsupported_kwargs(accepts_all) is accepts_all


def test_patch_init_state_kwargs_wraps_model_method():
    predictor = SimpleNamespace(model=SimpleNamespace(init_state=init_state))
    patch_init_state_kwargs(predictor)
    assert predictor.model.init_state(resource_path="x", offload_state_to_cpu=False) == ("x", False)
