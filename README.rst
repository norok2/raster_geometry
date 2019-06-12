Raster Geometry
===============

**Raster Geometry** - Create/manipulate N-dim raster geometric shapes.


Overview
--------

This software provides a library for generating or otherwise manipulating
N-dimensional raster geometric shapes.

Most of the code is used is used in a number of projects where it is tested
against real-life scenarios.

As a result of the code maturity, some of the library components may
undergo (eventually heavy) refactoring.
While this is not expected, this will be documented.
Please file a bug report if you detect an undocumented refactoring.

Releases information are available through ``NEWS.rst``.

For a more comprehensive list of changes see ``CHANGELOG.rst`` (automatically
generated from the version control system).


Features
--------

The 2D geometrical shapes currently available are:

 - square
 - rectangle
 - rhombus
 - circle
 - ellipse

The 3D geometrical shapes currently available are:

 - cube
 - cuboid
 - rhomboid
 - sphere
 - ellipsoid
 - cylinder

The N-dim geometrical shapes currently available are:

 - cuboid: sum[abs(x_n/a_n)^inf] < 1
 - superellipsoid: sum[abs(x_n/a_n)^k] < 1
 - prism: stack (N-1)-D rendered objects on given axis

etc.

Additional may be added in the future.


Installation
------------

The recommended way of installing the software is through
`PyPI <https://pypi.python.org/pypi/raster_geometry>`__:

.. code:: bash

    $ pip install raster_geometry

Alternatively, you can clone the source repository from
`Bitbucket <https://bitbucket.org/norok2/raster_geometry>`__:

.. code:: bash

    $ git clone git@bitbucket.org:norok2/raster_geometry.git
    $ cd raster_geometry
    $ pip install -e .

For more details see also ``INSTALL.rst``.


License
-------

This work is licensed through the terms and conditions of the
`GPLv3+ <http://www.gnu.org/licenses/gpl-3.0.html>`__ See the
accompanying ``LICENSE.rst`` for more details.


Acknowledgements
----------------

For a complete list of authors please see ``AUTHORS.rst``.

People who have influenced this work are acknowledged in ``THANKS.rst``.

