==========
harmonic-alignment
==========

.. image:: https://api.travis-ci.com/KrishnaswamyLab/harmonic-alignment.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/harmonic-alignment
    :alt: Travis CI Build
.. image:: https://coveralls.io/repos/github/KrishnaswamyLab/harmonic-alignment/badge.svg?branch=master
    :target: https://coveralls.io/github/KrishnaswamyLab/harmonic-alignment?branch=master
    :alt: Coverage Status
.. image:: https://img.shields.io/twitter/follow/scottgigante.svg?style=social&label=Follow
    :target: https://twitter.com/scottgigante
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/scottgigante/tasklogger.svg?style=social&label=Stars
    :target: https://github.com/scottgigante/tasklogger/
    :alt: GitHub stars

Installation
------------

harmonic-alignment is not yet available on `pip`. Install by running the following in a terminal::

    pip install --user git+https://github.com/KrishnaswamyLab/harmonic-alignment#subdirectory=python

Usage
-----

You can use harmonic-alignment as follows::

    import harmonicalignment
    ha_op = harmonicalignment.HarmonicAlignment(n_filters=4)
    ha_op.align(X, Y)
    XY_aligned = ha_op.diffusion_map()
