"""LLM backends: in-process MLX, or an upstream OpenAI-compatible server."""
from .. import config


def get():
    """The active backend module. CLOXY_LLM_URL set = upstream (e.g. an MLX cluster)."""
    if config.LLM_URL:
        from . import openai_backend
        return openai_backend
    from . import mlx_backend
    return mlx_backend
