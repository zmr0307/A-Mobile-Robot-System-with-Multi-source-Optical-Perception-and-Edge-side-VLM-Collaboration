# WL100 Bringup — 一键启动全部节点
# TODO: 明天完善具体实现

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera', default_value='false',
        description='是否启动 D435i 相机'
    )

    # FAST-LIO 定位模式下设为 false，由 fastlio2 接管 odom→base_footprint TF
    publish_tf_arg = DeclareLaunchArgument(
        'publish_chassis_tf', default_value='true',
        description='是否发布轮式里程计 TF (odom→base_footprint)'
    )

    # ---- 1. 机器人描述 (URDF + TF 树) ----
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('wl100_description'),
                'launch', 'display.launch.py'
            )
        )
    )

    # ---- 2. 底盘串口通信节点 ----
    chassis_node = Node(
        package='wl100_teleop',
        executable='serial_node',
        name='wl100_serial_node',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('wl100_bringup'),
                'config', 'chassis_params.yaml'
            ),
            {'publish_tf': LaunchConfiguration('publish_chassis_tf')},
        ],
    )

    # ---- 3. 雷达节点 ----
    lidar_node = Node(
        package='unitree_lidar_ros2',
        executable='unitree_lidar_ros2_node',
        name='unitree_lidar_ros2_node',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('wl100_bringup'),
                'config', 'lidar_params.yaml'
            )
        ],
    )

    # ---- 4. D435i 相机（可选）----
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py'
            )
        ),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'enable_color': 'true',
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'rgb_camera.color_profile': '640x480x30',
            'publish_tf': 'true',
            'tf_publish_rate': '0.0',
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
    )

    return LaunchDescription([
        launch_camera_arg,
        publish_tf_arg,
        description_launch,
        chassis_node,
        lidar_node,
        camera_launch,
    ])
