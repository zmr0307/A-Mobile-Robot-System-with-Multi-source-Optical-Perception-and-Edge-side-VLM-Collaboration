#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDL → FAST-LIO Localizer 桥接节点

工作流程:
  1. 启动后触发 HDL 的 /relocalize (BBS 全局定位)
  2. 等待 HDL 发布 /hdl_localization/relocalize_succeeded 成功事件
  3. 监听 HDL 发布的 /hdl_localization/pose，等待获得稳定的当前轮全局位姿
  4. 将该位姿通过 /localizer/relocalize Service 传给 localizer
  5. 等待 localizer 确认重定位成功
  6. 打印成功信息，节点进入空转（桥接任务完成）

使用方法:
  ros2 run wl100_bringup hdl_to_localizer_bridge.py
"""

import math
from collections import deque
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from interface.srv import Relocalize, IsValid


class HdlToLocalizerBridge(Node):
    def __init__(self):
        super().__init__('hdl_to_localizer_bridge')

        # 参数
        self.declare_parameter('pcd_path', '/home/nvidia/robot_ws/PCD/fastlio_scans.pcd')
        self.declare_parameter('auto_relocalize', True)
        self.declare_parameter('hdl_odom_timeout', 60.0)  # 等待 HDL 定位结果的超时秒数
        self.declare_parameter('hdl_pose_window_size', 12)  # HDL 位姿滑动窗口大小
        self.declare_parameter('hdl_pose_std_thresh', 0.025)  # HDL XY 标准差阈值 (m)
        self.declare_parameter('hdl_z_std_thresh', 0.02)  # HDL Z 标准差阈值 (m)
        self.declare_parameter('hdl_yaw_std_thresh_deg', 0.5)  # HDL 航向标准差阈值 (deg)

        self.pcd_path = self.get_parameter('pcd_path').get_parameter_value().string_value
        self.auto_relocalize = self.get_parameter('auto_relocalize').get_parameter_value().bool_value
        self.hdl_odom_timeout = self.get_parameter('hdl_odom_timeout').get_parameter_value().double_value
        self.hdl_pose_window_size = (
            self.get_parameter('hdl_pose_window_size').get_parameter_value().integer_value
        )
        self.hdl_pose_std_thresh = (
            self.get_parameter('hdl_pose_std_thresh').get_parameter_value().double_value
        )
        self.hdl_z_std_thresh = (
            self.get_parameter('hdl_z_std_thresh').get_parameter_value().double_value
        )
        self.hdl_yaw_std_thresh = math.radians(
            self.get_parameter('hdl_yaw_std_thresh_deg').get_parameter_value().double_value
        )

        # 状态
        self._hdl_pose_received = False
        self._hdl_relocalize_succeeded = False
        self._localizer_initialized = False
        self._localizer_request_inflight = False
        self._localizer_relocalize_sent = False
        self._hdl_position = None
        self._hdl_orientation = None
        self._hdl_odom_sub = None
        self._localization_check_timer = None
        self._hdl_pose_window = deque(maxlen=max(1, self.hdl_pose_window_size))

        # HDL relocalize client
        self._hdl_reloc_client = self.create_client(Empty, '/relocalize')

        # Localizer relocalize client
        self._localizer_reloc_client = self.create_client(Relocalize, '/localizer/relocalize')
        self._localizer_check_client = self.create_client(IsValid, '/localizer/relocalize_check')

        self._hdl_success_sub = self.create_subscription(
            Bool,
            '/hdl_localization/relocalize_succeeded',
            self._hdl_relocalize_succeeded_callback,
            10
        )

        self.get_logger().info('🔗 HDL → Localizer 桥接节点已启动')
        self.get_logger().info(f'   PCD 地图路径: {self.pcd_path}')

        if self.auto_relocalize:
            # 延迟 3 秒后自动触发 HDL 全局定位（等传感器启动）
            self._trigger_timer = self.create_timer(3.0, self._trigger_hdl_relocalize)
        else:
            self.get_logger().info('   手动模式：请调用 /relocalize 或使用 RViz 的 2D Pose Estimate 触发全局定位')

        # 监听 RViz 的 2D Pose Estimate
        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._initialpose_callback, 10
        )

    def _trigger_hdl_relocalize(self):
        """触发 HDL 的 BBS 全局定位"""
        self._trigger_timer.cancel()  # 只触发一次

        self._hdl_relocalize_succeeded = False
        self._hdl_pose_received = False
        self._hdl_position = None
        self._hdl_orientation = None
        self._hdl_pose_window.clear()

        if not self._hdl_reloc_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('❌ HDL /relocalize 服务不可用，请确认 HDL 已启动')
            return

        self.get_logger().info('🌍 正在触发 HDL BBS 全局定位...')
        request = Empty.Request()
        future = self._hdl_reloc_client.call_async(request)
        future.add_done_callback(self._hdl_reloc_done)

        # 设置超时检查
        self._timeout_timer = self.create_timer(
            self.hdl_odom_timeout, self._check_timeout
        )

    def _hdl_reloc_done(self, future):
        """HDL relocalize Service 返回"""
        try:
            future.result()
            self.get_logger().info('✅ HDL BBS 全局定位已触发，等待 HDL 多帧稳定确认成功事件...')
        except Exception as e:
            self.get_logger().error(f'❌ HDL 全局定位调用失败: {e}')

    def _hdl_relocalize_succeeded_callback(self, msg: Bool):
        """收到 HDL 当前轮重定位稳定确认成功事件"""
        if not msg.data:
            return

        if self._localizer_initialized or self._localizer_request_inflight or self._localizer_relocalize_sent:
            return

        self._hdl_relocalize_succeeded = True
        self._hdl_pose_received = False
        self._hdl_position = None
        self._hdl_orientation = None
        self._hdl_pose_window.clear()
        self.get_logger().info('✅ HDL 多帧稳定确认已通过，开始采集当前轮 HDL 位姿窗口...')
        self._start_listening_hdl()

    def _start_listening_hdl(self):
        """开始监听 HDL 真实的全局位姿"""
        if self._hdl_odom_sub is not None:
            return
        self.get_logger().info('📡 开始监听真实的 HDL 全局定位结果...')
        self._hdl_odom_sub = self.create_subscription(
            Odometry, '/hdl_localization/pose', self._hdl_odom_callback, 10
        )

    def _hdl_odom_callback(self, msg: Odometry):
        """接收 HDL 发布的位姿"""
        if self._localizer_initialized or self._localizer_request_inflight or self._localizer_relocalize_sent:
            return  # 已完成桥接，忽略后续消息

        if not self._hdl_relocalize_succeeded:
            return

        if not self._hdl_pose_received:
            self._hdl_pose_received = True

        # 提取位姿
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self._hdl_position = (p.x, p.y, p.z)
        self._hdl_orientation = (q.x, q.y, q.z, q.w)

        # 从四元数提取 yaw/pitch/roll
        yaw, pitch, roll = self._quat_to_euler(q.x, q.y, q.z, q.w)

        current_pose = (p.x, p.y, p.z, yaw, pitch, roll)
        self._hdl_pose_window.append(current_pose)

        if len(self._hdl_pose_window) < self.hdl_pose_window_size:
            self.get_logger().info(
                f'📍 HDL 位姿窗口填充 {len(self._hdl_pose_window)}/{self.hdl_pose_window_size}: '
                f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} yaw={math.degrees(yaw):.1f}°'
            )
            return

        mean_pose, pos_std, yaw_std = self._compute_window_stats()
        self.get_logger().info(
            f'📍 HDL 位姿窗口统计: '
            f'mean=({mean_pose[0]:.3f}, {mean_pose[1]:.3f}, {mean_pose[2]:.3f}) '
            f'yaw={math.degrees(mean_pose[3]):.1f}° '
            f'σx={pos_std[0] * 100.0:.1f}cm σy={pos_std[1] * 100.0:.1f}cm '
            f'σz={pos_std[2] * 100.0:.1f}cm '
            f'σyaw={math.degrees(yaw_std):.1f}°'
        )

        if (
            pos_std[0] > self.hdl_pose_std_thresh or
            pos_std[1] > self.hdl_pose_std_thresh or
            pos_std[2] > self.hdl_z_std_thresh or
            yaw_std > self.hdl_yaw_std_thresh
        ):
            return

        # 将位姿传给 localizer
        self._send_to_localizer(
            mean_pose[0], mean_pose[1], mean_pose[2], mean_pose[3], mean_pose[4], mean_pose[5]
        )

    def _initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """接收 RViz 发布的 2D Pose Estimate"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw, pitch, roll = self._quat_to_euler(q.x, q.y, q.z, q.w)
        
        self.get_logger().info(
            f'👉 收到 RViz 手动设定初始位姿: x={p.x:.3f} y={p.y:.3f} '
            f'yaw={math.degrees(yaw):.1f}°'
        )
        self._send_to_localizer(p.x, p.y, 0.0, yaw, 0.0, 0.0, force=True)

    def _send_to_localizer(self, x, y, z, yaw, pitch, roll, force=False):
        """调用 localizer 的 relocalize Service"""
        if self._localizer_request_inflight:
            self.get_logger().warn('⚠️ Localizer 重定位请求正在进行中，本次初始位姿未发送')
            return

        if not force and (self._localizer_initialized or self._localizer_relocalize_sent):
            return

        if force:
            self.get_logger().warn('🧭 手动初始位姿强制转发 Localizer，将覆盖当前自动重定位状态')
            self._hdl_relocalize_succeeded = False
            self._localizer_initialized = False
            self._localizer_relocalize_sent = False
            self._hdl_pose_window.clear()
            if self._localization_check_timer is not None:
                self._localization_check_timer.cancel()
                self._localization_check_timer = None

        if not self._localizer_reloc_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('❌ Localizer /localizer/relocalize 服务不可用')
            return

        self._localizer_request_inflight = True
        self._localizer_relocalize_sent = True

        request = Relocalize.Request()
        request.pcd_path = self.pcd_path
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.yaw = float(yaw)
        request.pitch = float(pitch)
        request.roll = float(roll)

        source = '手动初始位姿' if force else 'HDL 全局定位'
        self.get_logger().info(f'📤 向 Localizer 发送{source}和 PCD 地图...')
        future = self._localizer_reloc_client.call_async(request)
        future.add_done_callback(self._localizer_reloc_done)

    def _localizer_reloc_done(self, future):
        """Localizer relocalize 返回"""
        self._localizer_request_inflight = False
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'✅ Localizer 重定位成功: {result.message}')
                self._localizer_initialized = True

                # 取消超时计时器
                if hasattr(self, '_timeout_timer'):
                    self._timeout_timer.cancel()

                # 延迟 2 秒后检查定位状态
                if self._localization_check_timer is None:
                    self._localization_check_timer = self.create_timer(2.0, self._check_localization)
            else:
                self.get_logger().error(f'❌ Localizer 重定位失败: {result.message}')
                self._localizer_initialized = False
                self._localizer_relocalize_sent = False
        except Exception as e:
            self.get_logger().error(f'❌ Localizer 重定位调用异常: {e}')
            self._localizer_initialized = False
            self._localizer_relocalize_sent = False

    def _check_localization(self):
        """检查 localizer 的定位是否收敛"""
        if not self._localizer_check_client.wait_for_service(timeout_sec=2.0):
            return

        request = IsValid.Request()
        request.code = 0
        future = self._localizer_check_client.call_async(request)
        future.add_done_callback(self._localization_check_done)

    def _localization_check_done(self, future):
        """定位检查结果"""
        try:
            result = future.result()
            if result.valid:
                self.get_logger().info('🎯 Localizer 定位已收敛！导航系统就绪。')
                if self._localization_check_timer is not None:
                    self._localization_check_timer.cancel()
                    self._localization_check_timer = None
            else:
                self.get_logger().warn('⏳ Localizer 定位尚未收敛，等待中...')
        except Exception as e:
            self.get_logger().error(f'定位检查异常: {e}')

    def _check_timeout(self):
        """超时检查"""
        if hasattr(self, '_timeout_timer'):
            self._timeout_timer.cancel()
            self._timeout_timer = None

        if not self._localizer_initialized:
            if not self._hdl_relocalize_succeeded:
                self.get_logger().error(
                    f'⏰ 超时 ({self.hdl_odom_timeout}s)！'
                    f'HDL 未发布当前轮稳定确认成功事件。请检查 HDL、全局地图和点云质量。'
                )
            else:
                self.get_logger().error(
                    f'⏰ 超时 ({self.hdl_odom_timeout}s)！'
                    f'HDL 已稳定确认，但 Localizer 重定位未完成。请检查 Localizer 服务和 PCD 地图。'
                )

    def _compute_window_stats(self):
        """计算窗口内 HDL 位姿均值和标准差"""
        xs = [p[0] for p in self._hdl_pose_window]
        ys = [p[1] for p in self._hdl_pose_window]
        zs = [p[2] for p in self._hdl_pose_window]
        yaws = [p[3] for p in self._hdl_pose_window]
        pitches = [p[4] for p in self._hdl_pose_window]
        rolls = [p[5] for p in self._hdl_pose_window]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        mean_z = sum(zs) / len(zs)
        mean_pitch = sum(pitches) / len(pitches)
        mean_roll = sum(rolls) / len(rolls)

        mean_sin = sum(math.sin(y) for y in yaws) / len(yaws)
        mean_cos = sum(math.cos(y) for y in yaws) / len(yaws)
        mean_yaw = math.atan2(mean_sin, mean_cos)

        std_x = self._std(xs, mean_x)
        std_y = self._std(ys, mean_y)
        std_z = self._std(zs, mean_z)
        yaw_errors = [self._angle_diff(y, mean_yaw) for y in yaws]
        std_yaw = math.sqrt(sum(err * err for err in yaw_errors) / len(yaw_errors))

        mean_pose = (mean_x, mean_y, mean_z, mean_yaw, mean_pitch, mean_roll)
        return mean_pose, (std_x, std_y, std_z), std_yaw

    @staticmethod
    def _std(values, mean):
        if not values:
            return 0.0
        return math.sqrt(sum((v - mean) * (v - mean) for v in values) / len(values))

    @staticmethod
    def _angle_diff(a, b):
        """返回角度差 (rad)，范围 [-pi, pi]"""
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff

    @staticmethod
    def _quat_to_euler(qx, qy, qz, qw):
        """四元数 → 欧拉角 (yaw, pitch, roll)"""
        # yaw (Z)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # pitch (Y)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # roll (X)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        return yaw, pitch, roll


def main(args=None):
    rclpy.init(args=args)
    node = HdlToLocalizerBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
