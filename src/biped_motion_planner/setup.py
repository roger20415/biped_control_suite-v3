from setuptools import find_packages, setup

package_name = 'biped_motion_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roger20415',
    maintainer_email='roger20415@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'biped_motion_planner_node = biped_motion_planner.biped_motion_planner_node:main',
            'swing_leg_control_node = biped_motion_planner.swing_leg_control_node:main',
            'stance_leg_control_node = biped_motion_planner.stance_leg_control_node:main',
            'counterweight_control_node = biped_motion_planner.counterweight_control_node:main',
        ],
    },
)
