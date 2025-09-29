from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='thallos_ros', # 💡 패키지 이름 변경
            executable='camera_publisher_node.py',
            name='camera_publisher_node',
            output='screen',
        ),

        Node(
            package='thallos_ros', # 💡 패키지 이름 변경
            executable='object_detection_processing_node.py',
            name='object_detection_processing_node',
            output='screen',
            remappings=[
                ('/camera/image_raw', '/lgh/camera/image_raw')
            ]
        ),

        Node(
            package='thallos_ros', # 💡 패키지 이름 변경
            executable='ui_display_node.py',
            name='ui_display_node',
            output='screen',
            remappings=[
                ('/processing/visualized_image', '/processing/visualized_image'),
                ('/processing/data_output', '/processing/data_output')
            ]
        )
    ])