"""Training-time real-time chunking scheduling support for OpenPI.

The model sampler uses hard action-prefix conditioning learned during training.
This package provides the deterministic runtime timeline; legacy inference-time
replacement, soft-mask, and VJP guidance are intentionally unavailable.
"""

from openpi.rtc.timeline import ActionPlan
from openpi.rtc.timeline import DispatchAction
from openpi.rtc.timeline import RTCController
from openpi.rtc.timeline import RTCRequest
from openpi.rtc.timeline import RTCStateError

__all__ = ["ActionPlan", "DispatchAction", "RTCController", "RTCRequest", "RTCStateError"]
