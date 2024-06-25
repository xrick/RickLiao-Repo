import sys
from os.path import join, dirname
from os import environ
from distutils.core import setup, Extension

module1 = Extension('myaudioop',
                    sources = ['audioop.c'])

setup (name = 'audioop',
       version = '1.0',
       description = 'Audioop module from Python 2.7 standard library',
       ext_modules = [module1])
