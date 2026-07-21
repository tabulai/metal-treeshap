"""TreeSHAP inference accelerated by Apple GPUs through Metal."""

from .explainer import MetalTreeExplainer

__all__ = ["MetalTreeExplainer"]
__version__ = "0.1.1"
