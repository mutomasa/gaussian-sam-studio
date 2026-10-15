"""チェックポイントの解決。

sam3 の `download_ckpt_from_hf` は revision を指定せず main を取りに行くため使わず、
`configs/model.yaml` で固定した revision の重みを HF キャッシュから取る（未取得ならダウンロード）。
"""

from __future__ import annotations

from collections.abc import Callable

SAM31_CHECKPOINT = "sam3.1_multiplex.pt"

Downloader = Callable[..., str]


def _hf_hub_download(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def resolve_checkpoint(
    checkpoint_path: str | None,
    repo_id: str,
    revision: str,
    filename: str = SAM31_CHECKPOINT,
    download: Downloader = _hf_hub_download,
) -> str:
    """明示パスがあればそれを、無ければ固定 revision の重みのローカルパスを返す。"""
    if checkpoint_path:
        return checkpoint_path
    if not repo_id or not revision:
        raise ValueError("checkpoint_path か sam31.source（repo_id と revision）を設定してください")
    return download(repo_id=repo_id, filename=filename, revision=revision)
