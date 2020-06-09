#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import re
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string
with open(path.join(here, 'crl_kino', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
    name='crl_kino',
    version=version,
    description='Compositional reinforcement learning based kinodynamic planning',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sldai/crl_kino',
    author='sldai',
    author_email='daishilong1236@gmail.com',
    license='MIT',
    python_requires='>=3.6',
    classifiers=[
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='reinforcement learning rrt planning',
    packages=find_packages(include=['crl_kino', 'crl_kino.*', 'tianshou', 'tianshou.*']),
    install_requires=[
        'gym>=0.15.0',
        'tqdm',
        'numpy',
        'tensorboard',
        'torch>=1.4.0',
        'matplotlib>=3.1.0',
        'transforms3d',
        'seaborn',
    ],
)
