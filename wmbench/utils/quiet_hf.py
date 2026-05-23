"""Disable Hugging Face / Transformers weight-loading progress spam (Kaggle/Jupyter)."""

from __future__ import annotations

import os


def quiet_hf_loading() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from transformers.utils import logging as tf_logging

        tf_logging.disable_progress_bar()
        tf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
