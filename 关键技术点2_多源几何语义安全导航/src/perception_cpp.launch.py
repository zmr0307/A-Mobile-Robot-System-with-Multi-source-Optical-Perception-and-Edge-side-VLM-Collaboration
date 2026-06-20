"""
WL100 C++ TensorRT 感知系统启动文件 — D435i 相机 + YOLOE-26 TRT C++ 节点

用法：
  # 完整启动（相机 + TRT C++ 推理）
  ros2 launch wl100_perception_cpp perception_cpp.launch.py

  # 仅启动推理（相机已由其他 launch 启动时）
  ros2 launch wl100_perception_cpp perception_cpp.launch.py launch_camera:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── 包路径 ──
    perception_dir = get_package_share_directory('wl100_perception_cpp')
    realsense_dir = get_package_share_directory('realsense2_camera')

    # ── Launch 参数 ──
    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera',
        default_value='true',
        description='是否同时启动 D435i 相机'
    )

    yolo_delay_arg = DeclareLaunchArgument(
        'yolo_delay',
        default_value='3.0',
        description='C++ 推理节点延迟启动秒数（等待相机初始化）'
    )

    # ── D435i 相机（带对齐深度）──
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'enable_sync': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x15',
            'publish_tf': 'true',
            'tf_publish_rate': '0.0',
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
    )

    # ── 延迟 20s 后强制启用点云 ──
    # realsense v4.56.3 内部参数名为 pointcloud__neon_.enable，
    # rs_launch.py 的 pointcloud.enable 无法正确映射，需用 param set 修复
    enable_pointcloud = TimerAction(
        period=20.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'param', 'set',
                     '/camera/camera', 'pointcloud__neon_.enable', 'true'],
                output='screen',
            ),
        ],
    )

    # ── YOLOE-26 TRT C++ 检测节点 ──
    yolo_trt_node = Node(
        package='wl100_perception_cpp',
        executable='yolo_trt_node',
        name='yolo_detector',
        output='screen',
        parameters=[
            os.path.join(perception_dir, 'config', 'perception_params.yaml'),
        ],
    )

    # 延迟启动：等待相机 topic 就绪
    delayed_yolo = TimerAction(
        period=LaunchConfiguration('yolo_delay'),
        actions=[yolo_trt_node],
    )

    return LaunchDescription([
        launch_camera_arg,
        yolo_delay_arg,
        camera_launch,
        enable_pointcloud,
        delayed_yolo,
    ])
