from setuptools import find_packages, setup

package_name = 'robo_hockey_controller'

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
    maintainer='ellie',
    maintainer_email='ellieluo2003@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    
    entry_points={
            'console_scripts': [
                'controller = robo_hockey_controller.controller:main',
                'hockey_swing_server = robo_hockey_controller.hockey_swing_server:main',
                'hockey_swing_client = robo_hockey_controller.hockey_swing_client:main'
            ],
        },
)
