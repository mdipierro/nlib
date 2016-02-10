#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='nlib',
      version='0.3',
      description=open("README.md").read(),
      author='Massimo DiPierro',
      author_email='massimo.dipierro@gmail.com',
      url='https://github.com/mdipierro/nlib',
      install_requires=[],
      py_modules=["nlib"],
      license= 'BSD',
      package_data = {'': ['README.md']},
      keywords='numerical',
     )

# vim:set shiftwidth=4 tabstop=4 expandtab:
