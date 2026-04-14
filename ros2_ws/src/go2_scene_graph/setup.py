from glob import glob

from setuptools import find_packages, setup

package_name = "go2_scene_graph"

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
    description="3D object-centric semantic scene graph builder for go2-semantic-nav.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scene_graph_node = go2_scene_graph.scene_graph_node:main",
        ],
    },
)
