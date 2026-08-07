"""Real-Time Chunking (RTC) support for openpi.

Reference: arXiv 2506.07339, "Real-Time Execution of Action Chunking Flow Policies".

RTC implements inference-time asynchronous action chunking with inpainting guidance
during flow/diffusion denoising. It modifies the flow sampling loop to guide the
denoised trajectory toward consistency with previously committed actions.
"""

from openpi.rtc.timeline import ActionPlan
from openpi.rtc.timeline import DispatchAction
from openpi.rtc.timeline import RTCController
from openpi.rtc.timeline import RTCRequest
from openpi.rtc.timeline import RTCStateError

__all__ = ["ActionPlan", "DispatchAction", "RTCController", "RTCRequest", "RTCStateError"]
