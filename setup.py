#!/usr/bin/python
'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
from __future__ import print_function

import subprocess
import os
import warnings
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from distutils.command.build import build
from setuptools.command.build_ext import build_ext
from os.path import join as pjoin
import sys

try:
    import pybind11
except ImportError as e:
    raise ImportError("Pybind11 not installed. Please install C++ binding package pybind11 before running DDFacet install. "
                        "You should not see this message unless you are not running pip install (19.x) -- run pip install!")


pkg='killMS'
__version__ = "3.1.0"
build_root=os.path.dirname(__file__)

def backend(compile_options):

    if compile_options is not None:
        print("Compiling extension libraries with user defined options: '%s'"%compile_options)
    else:
        compile_options = ""
    
    compile_options += " -DENABLE_PYTHON_2=OFF "
    compile_options += " -DENABLE_PYTHON_3=ON "
    
    path = pjoin(build_root, pkg, 'cbuild')
    try:
        subprocess.check_call(["mkdir", path])
    except:
        warnings.warn("%s already exists in your source folder. We will not create a fresh build folder, but you "
                      "may want to remove this folder if the configuration has changed significantly since the "
                      "last time you run setup.py" % path)
    subprocess.check_call(["cd %s && cmake %s .. && make" %
                           (path, compile_options if compile_options is not None else ""), ""], shell=True)

class custom_install(install):
    install.user_options = install.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        install.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        install.run(self)

class custom_build(build):
    build.user_options = build.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        build.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        build.run(self)

class custom_build_ext(build_ext):
    build.user_options = build.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        build_ext.run(self)

class custom_sdist(sdist):
    def run(self):
        bpath = pjoin(build_root, pkg, 'cbuild')
        if os.path.isdir(bpath):
            subprocess.check_call(["rm", "-rf", bpath])
        sdist.run(self)

def define_scripts():
    #these must be relative to setup.py according to setuputils
    killms_scripts = [os.path.join(pkg, script_name) for script_name in ['kMS.py']]
    return killms_scripts

def readme():
    """ Return README contents """
    with open('README.md') as f:
        return f.read()

def requirements():
    install_requirements = [
        "DDFacet >= 0.7.0; python_version >= '3'",
        "bdsf > 1.8.15; python_version >= '3'"
    ]

    return install_requirements

setup(name=pkg,
      version=__version__,
      description='A Wirtinger-based direction-dependent radio interferometric calibration package',
      long_description = readme(),
      url='http://github.com/saopicc/killMS',
      classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Cyril Tasse',
      author_email='cyril.tasse@obspm.fr',
      license='GNU GPL v2',
      cmdclass={'install': custom_install,
                'build': custom_build,
                'sdist': custom_sdist,
                'build_ext': custom_build_ext
               },
      python_requires='>=3.0,<3.9',
      packages=[pkg],
      install_requires=requirements(),
      include_package_data=True,
      zip_safe=False,
      long_description_content_type='text/markdown',
      scripts=define_scripts(),
      extras_require={
          'fits-beam-support': ['meqtrees-cattery'],
      }
)
