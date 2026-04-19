"""
MDAM Decoder facade.
"""

# Cache moved to common.AttentionDecoderCache
# _decode_probs moved to common.select_action
from .decoder import MDAMDecoder as MDAMDecoder

__all__ = ["MDAMDecoder"]
