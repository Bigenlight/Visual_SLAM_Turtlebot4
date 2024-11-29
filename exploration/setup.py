from setuptools import setup
import os
from glob import glob

package_name = 'exploration'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Install package.xml
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Include launch files if any
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include package.xml
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='theo',
    maintainer_email='tpingouin@gmail.com',
    description='Frontier-based exploration package for TurtleBot4.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exploration_node = exploration.exploration:main',
            'exploration_kwi = exploration.exploration_kwi:main',
            'test_node = exploration.test:main',
            'exploration2_0 = exploration2_0.exploration2_0:main',
        ],
    },
)
