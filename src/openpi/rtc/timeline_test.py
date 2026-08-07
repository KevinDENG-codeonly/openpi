"""Tests for the deterministic RTC action timeline."""

import numpy as np
import pytest

from openpi.rtc import ActionPlan
from openpi.rtc import RTCController
from openpi.rtc import RTCRequest
from openpi.rtc import RTCStateError


def make_plan(
    *, generation_tick: int = 10, horizon: int = 8, action_dim: int = 2, robot_dim: int = 3
) -> ActionPlan:
    return ActionPlan(
        generation_tick=generation_tick,
        model_actions=np.arange(horizon * action_dim, dtype=np.float32).reshape(horizon, action_dim),
        robot_actions=np.arange(horizon * robot_dim, dtype=np.float32).reshape(horizon, robot_dim),
    )


def test_start_request_uses_exact_shifted_prefix():
    plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=4)
    controller.install_plan(plan)

    request = controller.start_request(current_tick=12, planned_delay_steps=2)

    assert request.start_tick == 12
    assert request.execution_horizon == 2
    assert request.planned_delay_steps == 2
    np.testing.assert_array_equal(request.action_prefix, plan.model_actions[2:4])


def test_start_request_preserves_an_empty_prefix_for_zero_delay():
    plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=4)
    controller.install_plan(plan)

    request = controller.start_request(current_tick=11, planned_delay_steps=0)

    assert request.execution_horizon == 1
    assert request.action_prefix.shape == (0, 2)
    np.testing.assert_array_equal(request.action_prefix, plan.model_actions[1:1])


def test_start_request_uses_s_min_when_it_exceeds_the_planned_delay():
    plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=3, training_max_delay_steps=4)
    controller.install_plan(plan)

    request = controller.start_request(current_tick=13, planned_delay_steps=1)

    assert request.execution_horizon == 3
    np.testing.assert_array_equal(request.action_prefix, plan.model_actions[3:4])


def test_start_request_rejects_delay_that_exhausts_the_remaining_horizon():
    plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=3, training_max_delay_steps=6)
    controller.install_plan(plan)

    with pytest.raises(RTCStateError, match="remaining action horizon"):
        controller.start_request(current_tick=16, planned_delay_steps=6)


@pytest.mark.parametrize(
    ("planned_delay_steps", "current_tick", "error_message"),
    [(-1, 11, "nonnegative"), (5, 15, "training range")],
)
def test_start_request_rejects_delay_outside_the_training_range(
    planned_delay_steps: int, current_tick: int, error_message: str
):
    plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=4)
    controller.install_plan(plan)

    with pytest.raises(RTCStateError, match=error_message):
        controller.start_request(current_tick=current_tick, planned_delay_steps=planned_delay_steps)


def test_start_request_requires_an_active_plan_and_only_one_inflight_request():
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=4)

    with pytest.raises(RTCStateError, match="active action plan"):
        controller.start_request(current_tick=0, planned_delay_steps=0)

    controller.install_plan(make_plan())
    request = controller.start_request(current_tick=11, planned_delay_steps=1)

    with pytest.raises(RTCStateError, match="already in flight"):
        controller.start_request(current_tick=11, planned_delay_steps=1)
    assert controller.inflight_request is request


def test_late_result_records_a_deadline_miss_and_keeps_the_old_plan():
    old_plan = make_plan(generation_tick=20)
    controller = RTCController(action_horizon=8, action_dim=2, s_min=2, training_max_delay_steps=4)
    controller.install_plan(old_plan)
    request = controller.start_request(current_tick=22, planned_delay_steps=2)
    late_plan = make_plan(generation_tick=22)

    with pytest.raises(RTCStateError, match="deadline miss"):
        controller.accept_result(request, late_plan, completion_tick=25)

    assert controller.active_plan is old_plan
    assert controller.inflight_request is None
    assert controller.deadline_miss_count == 1
    assert controller.consecutive_deadline_misses == 1


def test_on_time_result_switches_after_a_frozen_prefix():
    old_plan = make_plan(generation_tick=10)
    controller = RTCController(action_horizon=8, action_dim=2, s_min=2, training_max_delay_steps=4)
    controller.install_plan(old_plan)
    request = controller.start_request(current_tick=13, planned_delay_steps=3)
    new_model_actions = np.vstack(
        [old_plan.model_actions[3:6], np.full((5, 2), 99.0, dtype=np.float32)]
    )
    new_plan = ActionPlan(
        generation_tick=13,
        model_actions=new_model_actions,
        robot_actions=np.full((8, 3), 77.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(new_plan.model_actions[:3], old_plan.model_actions[3:6])
    controller.accept_result(request, new_plan, completion_tick=15)

    assert controller.active_plan is new_plan
    assert controller.inflight_request is None
    assert controller.consecutive_deadline_misses == 0
    dispatch = controller.action_for_tick(15)
    assert dispatch.kind == "action"
    np.testing.assert_array_equal(dispatch.model_action, old_plan.model_actions[5])
    np.testing.assert_array_equal(dispatch.robot_action, new_plan.robot_actions[2])


def test_accept_result_rejects_stale_requests_and_wrong_generation_ticks():
    old_plan = make_plan()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=4)
    controller.install_plan(old_plan)
    request = controller.start_request(current_tick=11, planned_delay_steps=1)
    stale_request = RTCRequest(
        request_id=request.request_id + 1,
        source_generation_tick=request.source_generation_tick,
        start_tick=request.start_tick,
        planned_delay_steps=request.planned_delay_steps,
        execution_horizon=request.execution_horizon,
        action_prefix=request.action_prefix,
    )

    with pytest.raises(RTCStateError, match="stale or mismatched request"):
        controller.accept_result(stale_request, make_plan(generation_tick=11), completion_tick=11)
    assert controller.inflight_request is request

    with pytest.raises(RTCStateError, match="generation_tick"):
        controller.accept_result(request, make_plan(generation_tick=12), completion_tick=11)
    assert controller.inflight_request is request
    assert controller.active_plan is old_plan


def test_action_for_tick_holds_when_the_active_plan_is_exhausted():
    controller = RTCController(action_horizon=2, action_dim=2, s_min=0, training_max_delay_steps=1)
    controller.install_plan(make_plan(generation_tick=5, horizon=2))

    dispatch = controller.action_for_tick(7)

    assert dispatch.kind == "hold"
    assert dispatch.model_action is None
    assert dispatch.robot_action is None


@pytest.mark.parametrize(
    ("generation_tick", "model_actions", "robot_actions"),
    [
        (-1, np.zeros((2, 2)), np.zeros((2, 3))),
        (0, np.zeros((2,)), np.zeros((2, 3))),
        (0, np.zeros((2, 2)), np.zeros((2,))),
        (0, np.zeros((2, 2)), np.zeros((3, 3))),
    ],
)
def test_action_plan_rejects_invalid_ticks_and_dimensions(
    generation_tick: int, model_actions: np.ndarray, robot_actions: np.ndarray
):
    with pytest.raises(RTCStateError):
        ActionPlan(
            generation_tick=generation_tick,
            model_actions=model_actions,
            robot_actions=robot_actions,
        )


@pytest.mark.parametrize(("action_horizon", "action_dim"), [(0, 2), (8, 0)])
def test_controller_rejects_invalid_dimensions(action_horizon: int, action_dim: int):
    with pytest.raises(RTCStateError):
        RTCController(
            action_horizon=action_horizon,
            action_dim=action_dim,
            s_min=0,
            training_max_delay_steps=1,
        )


def test_controller_validates_plan_horizon_and_model_dimension_on_install():
    controller = RTCController(action_horizon=8, action_dim=2, s_min=0, training_max_delay_steps=1)

    with pytest.raises(RTCStateError, match="action_horizon"):
        controller.install_plan(make_plan(horizon=7))
    with pytest.raises(RTCStateError, match="action_dim"):
        controller.install_plan(make_plan(action_dim=3))


def test_only_accepted_robot_acknowledgements_advance_the_logical_tick():
    old_plan = make_plan(generation_tick=0)
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=2)
    controller.install_plan(old_plan)
    request = controller.start_request(current_tick=1, planned_delay_steps=0)
    controller.accept_result(request, make_plan(generation_tick=1), completion_tick=1)
    controller.action_for_tick(1)

    assert controller.accepted_tick == 0
    assert controller.record_accepted_tick(acknowledged=False) == 0
    assert controller.accepted_tick == 0
    assert controller.record_accepted_tick(acknowledged=True) == 1
    assert controller.accepted_tick == 1
