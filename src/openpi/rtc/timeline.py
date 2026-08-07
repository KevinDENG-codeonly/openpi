"""Deterministic, hardware-independent action timeline for training-time RTC."""

from __future__ import annotations

import dataclasses
import numbers
from typing import Literal

import numpy as np


class RTCStateError(RuntimeError):
    """Raised when an RTC timeline transition would produce invalid state."""


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 0:
        raise RTCStateError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value <= 0:
        raise RTCStateError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclasses.dataclass(frozen=True)
class ActionPlan:
    """Model and robot action chunks generated for a single logical tick."""

    generation_tick: int
    model_actions: np.ndarray
    robot_actions: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation_tick", _nonnegative_integer(self.generation_tick, "generation_tick"))
        if not isinstance(self.model_actions, np.ndarray) or self.model_actions.ndim != 2:
            raise RTCStateError("model_actions must be a rank-2 numpy array")
        if not isinstance(self.robot_actions, np.ndarray) or self.robot_actions.ndim != 2:
            raise RTCStateError("robot_actions must be a rank-2 numpy array")
        if self.model_actions.shape[0] != self.robot_actions.shape[0]:
            raise RTCStateError("model_actions and robot_actions must have matching horizons")


@dataclasses.dataclass(frozen=True)
class RTCRequest:
    """A single inference request tied to the action plan that created it."""

    request_id: int
    source_generation_tick: int
    start_tick: int
    planned_delay_steps: int
    execution_horizon: int
    action_prefix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _nonnegative_integer(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "source_generation_tick",
            _nonnegative_integer(self.source_generation_tick, "source_generation_tick"),
        )
        object.__setattr__(self, "start_tick", _nonnegative_integer(self.start_tick, "start_tick"))
        object.__setattr__(
            self,
            "planned_delay_steps",
            _nonnegative_integer(self.planned_delay_steps, "planned_delay_steps"),
        )
        object.__setattr__(
            self,
            "execution_horizon",
            _nonnegative_integer(self.execution_horizon, "execution_horizon"),
        )
        if not isinstance(self.action_prefix, np.ndarray) or self.action_prefix.ndim != 2:
            raise RTCStateError("action_prefix must be a rank-2 numpy array")
        if self.action_prefix.shape[0] != self.planned_delay_steps:
            raise RTCStateError("action_prefix horizon must match planned_delay_steps")


@dataclasses.dataclass(frozen=True)
class DispatchAction:
    """The action to dispatch for one tick, or an explicit hold when unavailable."""

    kind: Literal["action", "hold"]
    model_action: np.ndarray | None = None
    robot_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("action", "hold"):
            raise RTCStateError(f"unknown dispatch kind {self.kind!r}")
        if self.kind == "action" and (self.model_action is None or self.robot_action is None):
            raise RTCStateError("action dispatches require model_action and robot_action")
        if self.kind == "hold" and (self.model_action is not None or self.robot_action is not None):
            raise RTCStateError("hold dispatches cannot include actions")


class RTCController:
    """Own the current RTC action plan and its single asynchronous replacement."""

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        s_min: int,
        training_max_delay_steps: int,
    ) -> None:
        self.action_horizon = _positive_integer(action_horizon, "action_horizon")
        self.action_dim = _positive_integer(action_dim, "action_dim")
        self.s_min = _nonnegative_integer(s_min, "s_min")
        self.training_max_delay_steps = _nonnegative_integer(
            training_max_delay_steps, "training_max_delay_steps"
        )
        self.active_plan: ActionPlan | None = None
        self.inflight_request: RTCRequest | None = None
        self.deadline_miss_count = 0
        self.consecutive_deadline_misses = 0
        self.accepted_tick = 0
        self._next_request_id = 0

    @property
    def current_plan(self) -> ActionPlan | None:
        """Return the plan currently used for dispatch."""
        return self.active_plan

    def install_plan(self, plan: ActionPlan) -> None:
        """Install an initial action plan when no replacement is in flight."""
        if self.inflight_request is not None:
            raise RTCStateError("cannot install an action plan while a request is in flight")
        self._validate_plan_dimensions(plan)
        self.active_plan = plan

    def start_request(self, current_tick: int, planned_delay_steps: int) -> RTCRequest:
        """Freeze the training-time prefix and begin one replacement request."""
        current_tick = _nonnegative_integer(current_tick, "current_tick")
        planned_delay_steps = _nonnegative_integer(planned_delay_steps, "planned_delay_steps")
        s = max(planned_delay_steps, self.s_min)

        if self.active_plan is None:
            raise RTCStateError("cannot start an RTC request without an active action plan")
        if self.inflight_request is not None:
            raise RTCStateError("an RTC request is already in flight")
        if current_tick != self.active_plan.generation_tick + s:
            raise RTCStateError("current_tick must equal active_plan.generation_tick + execution_horizon")
        if planned_delay_steps > self.training_max_delay_steps:
            raise RTCStateError("planned_delay_steps is outside the training range")
        if planned_delay_steps > self.action_horizon - s:
            raise RTCStateError("planned_delay_steps exceeds the remaining action horizon")

        request = RTCRequest(
            request_id=self._next_request_id,
            source_generation_tick=self.active_plan.generation_tick,
            start_tick=current_tick,
            planned_delay_steps=planned_delay_steps,
            execution_horizon=s,
            action_prefix=self.active_plan.model_actions[s : s + planned_delay_steps].copy(),
        )
        self._next_request_id += 1
        self.inflight_request = request
        return request

    def accept_result(self, request: RTCRequest, result_plan: ActionPlan, completion_tick: int) -> None:
        """Accept an on-time result for the current request or record a deadline miss."""
        completion_tick = _nonnegative_integer(completion_tick, "completion_tick")
        if self.inflight_request is None:
            raise RTCStateError("no RTC request is in flight")
        if request is not self.inflight_request:
            raise RTCStateError("stale or mismatched request cannot complete")

        actual_delay = completion_tick - request.start_tick
        if actual_delay > request.planned_delay_steps:
            self.inflight_request = None
            self.deadline_miss_count += 1
            self.consecutive_deadline_misses += 1
            raise RTCStateError("RTC deadline miss: result exceeded its planned delay")
        if result_plan.generation_tick != request.start_tick:
            raise RTCStateError("result_plan.generation_tick must equal request.start_tick")

        self._validate_plan_dimensions(result_plan)
        self.active_plan = result_plan
        self.inflight_request = None
        self.consecutive_deadline_misses = 0

    def action_for_tick(self, tick: int) -> DispatchAction:
        """Return the current plan's action at ``tick``, or an explicit hold."""
        tick = _nonnegative_integer(tick, "tick")
        if self.active_plan is None:
            return DispatchAction(kind="hold")

        index = tick - self.active_plan.generation_tick
        if index < 0 or index >= self.action_horizon:
            return DispatchAction(kind="hold")
        return DispatchAction(
            kind="action",
            model_action=self.active_plan.model_actions[index],
            robot_action=self.active_plan.robot_actions[index],
        )

    def record_accepted_tick(self, *, acknowledged: bool) -> int:
        """Record a robot acknowledgement and advance only when it was accepted."""
        if acknowledged:
            self.accepted_tick += 1
        return self.accepted_tick

    def _validate_plan_dimensions(self, plan: ActionPlan) -> None:
        if not isinstance(plan, ActionPlan):
            raise RTCStateError("plan must be an ActionPlan")
        if plan.model_actions.shape[0] != self.action_horizon:
            raise RTCStateError("plan action_horizon does not match the controller")
        if plan.model_actions.shape[1] != self.action_dim:
            raise RTCStateError("plan action_dim does not match the controller")
