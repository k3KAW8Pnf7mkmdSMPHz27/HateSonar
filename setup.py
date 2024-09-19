#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys

from setuptools import find_packages, setup

# Package meta-data.
NAME = "hatesonar-onnxruntime"
DESCRIPTION = (
    "Fork of Hironsan's Hate Speech Detection Library, HateSonar, using onnxruntime."
)
URL = "https://github.com/k3KAW8Pnf7mkmdSMPHz27/HateSonar"
EMAIL = "jonatan.asketorp@gmail.com"
AUTHOR = "Jonatan Asketorp"
LICENSE = "MIT"

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()

required = ["numpy~=1.24.3", "onnxruntime~=1.19.2"]

setup(
    name=NAME,
    version="0.0.8",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=required,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
