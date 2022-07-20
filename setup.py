#!/usr/bin/env python

import os

import setuptools

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + "/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="catalystGA",
    version="0",
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=["catalystGA"],
)
