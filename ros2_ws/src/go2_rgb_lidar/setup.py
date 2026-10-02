from glob import glob

from setuptools import find_packages, setup

package_name = "go2_rgb_lidar"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf Guenena",
    maintainer_email="yusuf@example.com",
    description="GO2 front camera driver and LiDAR-projected aligned depth for the RGB-only GO2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "front_camera_node = go2_rgb_lidar.front_camera_node:main",
            "lidar_depth_node = go2_rgb_lidar.lidar_depth_node:main",
        ],
    },
)
