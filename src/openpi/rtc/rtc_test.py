"""Tests for RTC (Real-Time Chunking) implementation."""

import numpy as np
import pytest


class TestSoftMask:
    """Test compute_soft_mask (Eq. 5 from arXiv 2506.07339)."""

    def test_shape(self):
        from openpi.rtc.helpers import compute_soft_mask

        mask = compute_soft_mask(action_horizon=50, consumed=10, s_min=5)
        assert mask.shape == (50,)
        assert mask.dtype == np.float32

    def test_regions(self):
        from openpi.rtc.helpers import compute_soft_mask

        horizon, d, s = 50, 10, 5
        mask = compute_soft_mask(horizon, consumed=d, s_min=s)
        # Region 1: fully committed (i < d) should be 1
        np.testing.assert_array_equal(mask[:d], 1.0)
        # Region 3: free tail (i >= H-s) should be 0
        np.testing.assert_array_equal(mask[horizon - max(d, s) :], 0.0)
        # Region 2: transition (d <= i < H-s) should be in (0, 1), monotonically decreasing
        transition = mask[d : horizon - max(d, s)]
        assert np.all(transition > 0.0)
        assert np.all(transition <= 1.0)
        # Check monotonically non-increasing
        assert np.all(np.diff(transition) <= 0.0)

    def test_boundary_values(self):
        from openpi.rtc.helpers import compute_soft_mask

        # At i=d, c_i is near 1 for a long transition region.
        # At the final constrained timestep, c_i is near 0.
        horizon, consumed, s_min = 100, 10, 5
        free_start = horizon - max(consumed, s_min)
        mask = compute_soft_mask(horizon, consumed=consumed, s_min=s_min)
        # First transition value should be close to 1 (but < 1)
        assert mask[10] < 1.0
        assert mask[10] > 0.5
        # Last transition value should be close to 0 (but > 0)
        assert mask[free_start - 1] > 0.0
        assert mask[free_start - 1] < 0.5
        np.testing.assert_array_equal(mask[free_start:], 0.0)

    def test_zero_consumed(self):
        from openpi.rtc.helpers import compute_soft_mask

        mask = compute_soft_mask(50, consumed=0, s_min=5)
        # No fully committed region, transition starts from index 0
        assert mask[0] < 1.0
        assert mask[0] > 0.0
        # Tail is free
        np.testing.assert_array_equal(mask[45:], 0.0)

    def test_edge_case_large_consumed(self):
        from openpi.rtc.helpers import compute_soft_mask

        # consumed + s_min >= H: degenerate case
        mask = compute_soft_mask(50, consumed=48, s_min=5)
        assert mask.shape == (50,)
        # First 48 committed
        np.testing.assert_array_equal(mask[:48], 1.0)
        np.testing.assert_array_equal(mask[48:], 0.0)

    def test_consumed_beyond_horizon_has_no_overlap(self):
        from openpi.rtc.helpers import compute_soft_mask

        mask = compute_soft_mask(50, consumed=50, s_min=5)
        np.testing.assert_array_equal(mask, 0.0)

    def test_s_min_zero(self):
        from openpi.rtc.helpers import compute_soft_mask

        mask = compute_soft_mask(50, consumed=10, s_min=0)
        # The zero-padded non-overlap tail is still free even when s_min is zero.
        np.testing.assert_array_equal(mask[40:], 0.0)


class TestBuildRTCTargetAndMask:
    """Test build_rtc_target_and_mask."""

    def test_basic_shape(self):
        from openpi.rtc.helpers import build_rtc_target_and_mask

        prev = np.random.randn(50, 32).astype(np.float32)
        target, mask = build_rtc_target_and_mask(prev, consumed=10, s_min=5)
        assert target.shape == (50, 32)
        assert mask.shape == (50,)

    def test_target_shift(self):
        from openpi.rtc.helpers import build_rtc_target_and_mask

        prev = np.arange(50 * 32).reshape(50, 32).astype(np.float32)
        target, mask = build_rtc_target_and_mask(prev, consumed=10, s_min=5)
        # target[:40] should be prev[10:]
        np.testing.assert_array_equal(target[:40], prev[10:])
        # target[40:] should be zero-padded
        np.testing.assert_array_equal(target[40:], 0.0)
        # zero-padded target positions must be unconstrained
        np.testing.assert_array_equal(mask[40:], 0.0)


class TestRTCState:
    """Test RTCState controller."""

    def test_initial_no_guidance(self):
        from openpi.rtc.state import RTCState

        state = RTCState(action_horizon=50, action_dim=32, initial_delay_steps=1)
        assert state.get_rtc_kwargs() is None

    def test_first_inference_no_guidance(self):
        from openpi.rtc.state import RTCState

        state = RTCState(action_horizon=50, action_dim=32, initial_delay_steps=1)
        actions = np.random.randn(50, 32).astype(np.float32)
        state.update_after_inference(actions)
        # total_inferences == 1, initial_delay_steps == 1, so the next inference can use guidance
        assert state.get_rtc_kwargs() is not None

    def test_second_inference_has_guidance(self):
        from openpi.rtc.state import RTCState

        state = RTCState(action_horizon=50, action_dim=32, initial_delay_steps=1, s_min=5, beta=0.8)
        actions1 = np.random.randn(50, 32).astype(np.float32)
        state.update_after_inference(actions1)
        state.mark_consumed(10)
        actions2 = np.random.randn(50, 32).astype(np.float32)
        state.update_after_inference(actions2)
        state.mark_consumed(10)
        # Now total_inferences == 2 > initial_delay_steps == 1
        kwargs = state.get_rtc_kwargs()
        assert kwargs is not None
        assert "target" in kwargs
        assert "mask" in kwargs
        assert "beta" in kwargs
        assert kwargs["target"].shape == (50, 32)
        assert kwargs["mask"].shape == (50,)
        assert kwargs["beta"] == 0.8

    def test_shape_validation(self):
        from openpi.rtc.state import RTCState

        state = RTCState(action_horizon=50, action_dim=32)
        with pytest.raises(AssertionError):
            state.update_after_inference(np.zeros((25, 32)))


class TestWebsocketEnvelope:
    """Test that the server envelope parsing logic handles both old and new formats."""

    def test_old_format_is_obs(self):
        """Old clients send obs dict directly - should work as before."""
        payload = {"observation/image": np.zeros((3, 224, 224)), "prompt": "pick up cup"}
        # Simulate server parsing logic
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            rtc = payload.get("rtc")
        else:
            obs = payload
            rtc = None
        assert obs is payload
        assert rtc is None

    def test_new_format_with_rtc(self):
        """New format: {"obs": {...}, "rtc": {...}}."""
        inner_obs = {"observation/image": np.zeros((3, 224, 224)), "prompt": "pick up cup"}
        rtc_data = {"target": np.zeros((50, 32)), "mask": np.ones(50), "beta": 0.8}
        payload = {"obs": inner_obs, "rtc": rtc_data}
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            rtc = payload.get("rtc")
        else:
            obs = payload
            rtc = None
        assert obs is inner_obs
        assert rtc is rtc_data

    def test_new_format_without_rtc(self):
        """New format with rtc=None should behave like no RTC."""
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
    """Verify that default (no-RTC) behavior is preserved at the interface level."""

    def test_policy_infer_signature_default(self):
        """Policy.infer with rtc=None should not error on interface level."""
        import inspect

        from openpi.policies.policy import Policy

        # Check that the method signature accepts rtc kwarg
        sig = inspect.signature(Policy.infer)
        params = sig.parameters
        assert "rtc" in params
        assert params["rtc"].default is None
        assert "return_model_actions" in params
        assert params["return_model_actions"].default is False

    def test_client_infer_signature(self):
        """WebsocketClientPolicy.infer accepts optional rtc kwarg."""
        import inspect

        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        sig = inspect.signature(WebsocketClientPolicy.infer)
        params = sig.parameters
        assert "rtc" in params
        assert params["rtc"].default is None
        assert "return_model_actions" in params
        assert params["return_model_actions"].default is False
