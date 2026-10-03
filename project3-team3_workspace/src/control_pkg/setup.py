import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools', 'numpy', 'qpsolvers', 'osqp', 'scipy'],
    zip_safe=True,
    maintainer='george',
    maintainer_email='george@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'qp_ik_node = control_pkg.qp_ik_node:main',
            'joint_target_tcp_bridge_node = control_pkg.joint_target_tcp_bridge_node:main',
            'visual_servoing_motion_node = control_pkg.visual_servoing_motion_node:main',
            'task_automation_node = control_pkg.task_automation_node:main',
        ],
    },
)
