from setuptools import setup

package_name = 'stem_lib'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name, 
        package_name+'/layers', 
        package_name+'/models',
        package_name+'/stdlib',
        package_name+'/stdlib/collections',
        package_name+'/stdlib/concurrent',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ken',
    maintainer_email='fivetwothreesix@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
