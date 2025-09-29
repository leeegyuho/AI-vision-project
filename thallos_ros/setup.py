from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'thallos_ros' # 💡 패키지 이름 변경

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 💡 launch 파일 설치 경로 추가 (새로운 launch 폴더)
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*_launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A ROS 2 package for vehicle perception and UI alerts.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # 💡 Python 실행 노드 (스크립트) 등록
    entry_points={
        'console_scripts': [
            'camera_publisher_node = thallos_ros.camera_publisher_node:main', # 💡 모듈 경로 변경
            'object_detection_processing_node = thallos_ros.object_detection_processing_node:main', # 💡 모듈 경로 변경
            'ui_display_node = thallos_ros.ui_display_node:main', # 💡 모듈 경로 변경
        ],
    },
)
