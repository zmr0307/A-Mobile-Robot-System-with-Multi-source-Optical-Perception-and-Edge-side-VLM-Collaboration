#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时 MPPI vy_std 调节节点。

策略:
  - 导航开始/默认状态: FollowPath.vy_std = 0.02
  - distance_remaining <= 0.35m: FollowPath.vy_std = 0.025
  - 目标成功/取消/失败: 恢复 FollowPath.vy_std = 0.02
  - 防误触发: 同一个 goal 必须先看到 distance_remaining >= 0.10m，
    才允许后续进入 near 模式；避免脚本刚启动时收到旧的 0.000m feedback。

直接运行:
  python3 src/wl100_bringup/scripts/dynamic_vy_std_guard.py
"""

import math

import rclpy
from rclpy.node import Node

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


class DynamicVyStdGuard(Node):
    def __init__(self):
        super().__init__("dynamic_vy_std_guard")

        self.declare_parameter("controller_node", "/controller_server")
        self.declare_parameter("parameter_name", "FollowPath.vy_std")
        self.declare_parameter("far_vy_std", 0.02)
        self.declare_parameter("near_vy_std", 0.025)
        self.declare_parameter("enter_distance", 0.5)
        self.declare_parameter("arm_distance", 0.10)
        self.declare_parameter(
            "feedback_topic", "/navigate_to_pose/_action/feedback"
        )
        self.declare_parameter("status_topic", "/navigate_to_pose/_action/status")

        self.controller_node = self.get_parameter("controller_node").value
        self.parameter_name = self.get_parameter("parameter_name").value
        self.far_vy_std = float(self.get_parameter("far_vy_std").value)
        self.near_vy_std = float(self.get_parameter("near_vy_std").value)
        self.enter_distance = float(self.get_parameter("enter_distance").value)
        self.arm_distance = float(self.get_parameter("arm_distance").value)
        feedback_topic = self.get_parameter("feedback_topic").value
        status_topic = self.get_parameter("status_topic").value

        service_name = self.controller_node.rstrip("/") + "/set_parameters"
        self.set_param_client = self.create_client(SetParameters, service_name)

        self.active_goal_id = None
        self.goal_armed = False
        self.near_mode = False
        self.current_value = None
        self.request_inflight = False
        self.pending_value = None
        self.pending_reason = ""

        self.feedback_sub = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            feedback_topic,
            self.feedback_callback,
            10,
        )
        self.status_sub = self.create_subscription(
            GoalStatusArray,
            status_topic,
            self.status_callback,
            10,
        )

        self.pending_timer = self.create_timer(0.5, self.process_pending)
        self.queue_set(self.far_vy_std, "startup/default")

        self.get_logger().info(
            f"dynamic vy_std guard started: far={self.far_vy_std:.3f}, "
            f"near={self.near_vy_std:.3f}, enter_distance={self.enter_distance:.2f}m, "
            f"arm_distance={self.arm_distance:.2f}m"
        )

    @staticmethod
    def goal_uuid(goal_id):
        return tuple(goal_id.uuid)

    def queue_set(self, value, reason):
        value = float(value)
        if self.current_value is not None and math.isclose(
            self.current_value, value, abs_tol=1e-6
        ):
            return
        if self.pending_value is not None and math.isclose(
            self.pending_value, value, abs_tol=1e-6
        ):
            return
        self.pending_value = value
        self.pending_reason = reason
        self.process_pending()

    def process_pending(self):
        if self.pending_value is None or self.request_inflight:
            return
        if not self.set_param_client.service_is_ready():
            self.get_logger().warn(
                f"waiting for {self.controller_node}/set_parameters ...",
                throttle_duration_sec=3.0,
            )
            return

        value = self.pending_value
        reason = self.pending_reason
        self.pending_value = None
        self.pending_reason = ""
        self.request_inflight = True

        parameter = Parameter(
            name=self.parameter_name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=float(value),
            ),
        )
        request = SetParameters.Request()
        request.parameters = [parameter]
        future = self.set_param_client.call_async(request)
        future.add_done_callback(
            lambda fut, requested=value, why=reason: self.set_done(
                fut, requested, why
            )
        )

    def set_done(self, future, requested, reason):
        self.request_inflight = False
        try:
            response = future.result()
            ok = bool(response.results and response.results[0].successful)
            if ok:
                self.current_value = requested
                self.get_logger().info(
                    f"set {self.parameter_name}={requested:.3f} ({reason})"
                )
            else:
                msg = response.results[0].reason if response.results else "no result"
                self.get_logger().error(
                    f"failed to set {self.parameter_name}={requested:.3f}: {msg}"
                )
                self.pending_value = requested
                self.pending_reason = reason
        except Exception as exc:
            self.get_logger().error(
                f"set_parameters call failed for {self.parameter_name}: {exc}"
            )
            self.pending_value = requested
            self.pending_reason = reason
        self.process_pending()

    def feedback_callback(self, msg):
        goal_id = self.goal_uuid(msg.goal_id)
        if goal_id != self.active_goal_id:
            self.active_goal_id = goal_id
            self.goal_armed = False
            self.near_mode = False
            self.queue_set(self.far_vy_std, "new navigation goal")

        distance = float(msg.feedback.distance_remaining)
        if not math.isfinite(distance):
            return

        if not self.goal_armed and distance >= self.arm_distance:
            self.goal_armed = True
            self.get_logger().info(
                f"goal armed after distance_remaining={distance:.3f}m "
                f">= {self.arm_distance:.2f}m"
            )

        if self.goal_armed and not self.near_mode and distance <= self.enter_distance:
            self.near_mode = True
            self.queue_set(
                self.near_vy_std,
                f"distance_remaining={distance:.3f}m <= {self.enter_distance:.2f}m",
            )

    def status_callback(self, msg):
        if self.active_goal_id is None:
            return

        terminal_statuses = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        }
        for status in msg.status_list:
            if self.goal_uuid(status.goal_info.goal_id) != self.active_goal_id:
                continue
            if status.status in terminal_statuses:
                self.active_goal_id = None
                self.goal_armed = False
                self.near_mode = False
                self.queue_set(self.far_vy_std, "navigation terminal status")
                return


def main(args=None):
    rclpy.init(args=args)
    node = DynamicVyStdGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.queue_set(node.far_vy_std, "shutdown restore")
        for _ in range(10):
            if node.pending_value is None and not node.request_inflight:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
