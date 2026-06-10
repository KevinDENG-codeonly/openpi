"""Real-Time Chunking (RTC) support for openpi.

Reference: arXiv 2506.07339, "Real-Time Execution of Action Chunking Flow Policies".

RTC implements inference-time asynchronous action chunking with inpainting guidance
during flow/diffusion denoising. It modifies the flow sampling loop to guide the
denoised trajectory toward consistency with previously committed actions.
"""
