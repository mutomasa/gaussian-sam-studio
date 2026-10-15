"""上流 sam3 の不具合に対する互換処理。上流で修正されたら削除する。"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any


def drop_unsupported_kwargs(func: Callable[..., Any]) -> Callable[..., Any]:
    """`func` のシグネチャに無いキーワード引数を捨てて呼ぶラッパー。"""
    parameters = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **{k: v for k, v in kwargs.items() if k in parameters})

    return wrapper


def patch_init_state_kwargs(predictor: Any) -> Any:
    """`start_session` が渡す `offload_state_to_cpu` で multiplex 版 `init_state` が落ちる問題の回避。

    facebookresearch/sam3#544（修正 PR #543）。本リポジトリ #7。
    """
    model = predictor.model
    model.init_state = drop_unsupported_kwargs(model.init_state)
    return predictor
