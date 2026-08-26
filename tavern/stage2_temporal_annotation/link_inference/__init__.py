"""Link inference: the typed cascade of Table tab:ann-cascade."""
from .tlink import TLinkInferrer
from .slink import SLinkInferrer
from .alink import ALinkInferrer
from .mlink import MLinkInferrer

__all__ = ["TLinkInferrer", "SLinkInferrer", "ALinkInferrer", "MLinkInferrer"]
