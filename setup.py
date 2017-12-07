#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numpy >= 1.11', 'pandas >= 0.18.0', 'scipy', 'xarray',
                    'statsmodels', 'matplotlib', 'numba', 'patsy', 'seaborn',
                    'holoviews', 'bokeh']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
        name='mst_decoder',
        version='0.3.1',
        license='GPL-3.0',
        description=('Non-parametric stimulus decoding from'
                     ' multiunit spiking activity'),
        author='Michael Adkins',
        author_email='adkin099@umn.edu',
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
        url='https://github.com/mikeza/mst-decoder',
      )

# original_author='Eric Denovellis',
# original_author_email='edeno@bu.edu',