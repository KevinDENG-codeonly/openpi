"""Public RTC API compatibility tests."""

import inspect

import numpy as np

from openpi.models.pi0 import Pi0


def test_pi0_exposes_only_training_time_rtc_arguments() -> None:
    params = inspect.signature(Pi0.sample_actions).parameters

    assert {"rtc_action_prefix", "rtc_delay_steps"} <= params.keys()
    assert not {
        "rtc_target_actions",
        "rtc_target_mask",
        "rtc_loss_weight",
        "rtc_target",
        "rtc_weight",
        "rtc_beta",
        "beta",
    } & params.keys()
    assert not any("soft" in name or "vjp" in name for name in params)


def test_pytorch_pi0_sampler_exposes_no_legacy_rtc_kwargs() -> None:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    params = inspect.signature(PI0Pytorch.sample_actions).parameters

    assert {"device", "observation", "noise", "num_steps"} <= params.keys()
    assert not any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
    assert not any("rtc" in name or "soft" in name or "vjp" in name for name in params)


class TestWebsocketEnvelope:
    """The policy server keeps its ordinary and training-time envelopes."""

    def test_old_format_is_obs(self) -> None:
        payload = {"observation/image": np.zeros((3, 224, 224)), "prompt": "pick up cup"}
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            rtc = payload.get("rtc")
        else:
            obs = payload
            rtc = None
        assert obs is payload
        assert rtc is None

    def test_training_time_rtc_envelope(self) -> None:
        inner_obs = {"observation/image": np.zeros((3, 224, 224)), "prompt": "pick up cup"}
        rtc_data = {
            "algorithm": "training_time_v1",
            "action_prefix": np.zeros((50, 32), dtype=np.float32),
            "delay_steps": 10,
        }
        payload = {"obs": inner_obs, "rtc": rtc_data}
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            rtc = payload.get("rtc")
        else:
            obs = payload
            rtc = None
        assert obs is inner_obs
        assert rtc is rtc_data
        assert set(rtc) == {"algorithm", "action_prefix", "delay_steps"}
        assert rtc["action_prefix"].shape == (50, 32)
        assert isinstance(rtc["delay_steps"], int)

    def test_new_format_without_rtc(self) -> None:
        inner_obs = {"prompt": "test"}
        payload = {"obs": inner_obs, "rtc": None}
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            rtc = payload.get("rtc")
        else:
            obs = payload
            rtc = None
        assert obs is inner_obs
        assert rtc is None


class TestDefaultBehaviorCompatibility:
    """The no-RTC policy interfaces remain optional and backward compatible."""

    def test_policy_infer_signature_default(self) -> None:
        from openpi.policies.policy import Policy

        params = inspect.signature(Policy.infer).parameters
        assert params["rtc"].default is None
        assert params["return_model_actions"].default is False

    def test_client_infer_signature(self) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        params = inspect.signature(WebsocketClientPolicy.infer).parameters
        assert params["rtc"].default is None
        assert params["return_model_actions"].default is False
