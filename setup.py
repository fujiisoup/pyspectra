#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re

# load version form _version.py
VERSIONFILE = "pyspectra/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(
    name="pyspectra",
    version=verstr,
    author="Keisuke Fujii",
    author_email="fujii@me.kyoto-u.ac.jp",
    description=("Python small library for spectroscopy"),
    license="BSD 3-clause",
    keywords="spectroscopy, atomic data, machine learning",
    url="http://github.com/fujiisoup/xrfac",
    include_package_data=True,
    ext_modules=[],
    packages=["pyspectra",],
    package_dir={"pyspectra": "pyspectra"},
    package_data={'pyspectra': ['pyspectra/data/*.npz']},
    py_modules=["pyspectra.__init__"],
    test_suite="tests",
    install_requires="""
        numpy>=1.11
        scipy>=1.00
        """,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
