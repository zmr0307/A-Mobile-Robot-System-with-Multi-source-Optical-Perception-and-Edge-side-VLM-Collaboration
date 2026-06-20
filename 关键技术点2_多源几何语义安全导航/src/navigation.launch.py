"""
Nav2 导航启动文件 — WL100 全向底盘专用
=============================================
本启动文件仅启动 Nav2 导航核心节点，不启动 AMCL。
全局定位由 hdl_localization 负责（需在另一个终端单独启动）。

启动顺序：
  终端 1: ros2 launch wl100_bringup robot.launch.py         # 底盘 + URDF + LiDAR
  终端 2: ros2 launch fdilink_ahrs ahrs_driver.launch.py    # N100 IMU
  终端 3: ros2 launch hdl_localization hdl_localization_wl100.launch.py  # 全局定位
  终端 4: ros2 launch wl100_bringup navigation.launch.py    # ← 本文件（Nav2 导航）
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- 路径 ----------
    bringup_dir = get_package_share_directory('wl100_bringup')
    nav2_params_file = os.path.join(bringup_dir, 'config', 'nav2_params.yaml')

    # ---------- Launch 参数 ----------
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='是否使用仿真时间')

    declare_autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='自动激活生命周期节点')

    declare_params_file = DeclareLaunchArgument(
        'params_file', default_value=nav2_params_file,
        description='Nav2 参数文件路径')

    # ---------- 生命周期托管的节点列表 ----------
    lifecycle_nodes = [
        'map_server',
        'controller_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'velocity_smoother',
    ]

    # ---------- 节点定义 ----------

    # 地图服务器 — 加载 2D 栅格地图
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}])

    # 局部控制器 — MPPI (Omni)
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')])

    # 全局规划器 — SmacPlanner2D（Cost-Aware A*）
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'use_sim_time': use_sim_time}])

    # 恢复行为服务器 — spin, backup, wait
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'use_sim_time': use_sim_time}])

    # 行为树导航器
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'use_sim_time': use_sim_time}])

    # 速度平滑器
    velocity_smoother_node = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),           # 从 controller_server 接收
            ('cmd_vel_smoothed', 'cmd_vel')])      # 平滑后发送到底盘

    # 生命周期管理器 — 统一管理所有节点的启停
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'autostart': autostart,
            'node_names': lifecycle_nodes,
            'bond_timeout': 4.0,
            'attempt_respawn_reconnection': True,
        }])

    # ---------- 组装 ----------
    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_autostart)
    ld.add_action(declare_params_file)

    ld.add_action(map_server_node)
    ld.add_action(controller_server_node)
    ld.add_action(planner_server_node)
    ld.add_action(behavior_server_node)
    ld.add_action(bt_navigator_node)
    ld.add_action(velocity_smoother_node)
    ld.add_action(lifecycle_manager_node)

    return ld
