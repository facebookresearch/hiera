# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup

setup(
    name="hiera-transformer",
    version="0.1.2",
    author="Chaitanya Ryali, Daniel Bolya",
    url="https://github.com/facebookresearch/hiera",
    description="A fast, powerful, and simple hierarchical vision transformer",
    install_requires=["torch>=1.8.1", "timm>=0.4.12", "tqdm"],
    packages=find_packages(exclude=("examples", "build")),
    license = 'CC BY-NC 4.0',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)