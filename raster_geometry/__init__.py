#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Geometry - Create/manipulate N-dim raster geometric shapes.
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# ======================================================================
# :: External Imports
# import flyingcircus as fc  # Everything you always wanted to have in Python*
from flyingcircus import msg, dbg, fmt, fmtm, elapsed, report, pkg_paths
from flyingcircus import run_doctests
from flyingcircus import VERB_LVL, VERB_LVL_NAMES, D_VERB_LVL
from flyingcircus import HAS_JIT, jit

# ======================================================================
# :: Version
from raster_geometry._version import __version__

# ======================================================================
# :: Project Details
INFO = {
    'name': 'Raster Geometry',
    'author': 'Raster Geometry developers',
    'contrib': (
        'Riccardo Metere <rick@metere.it>',
    ),
    'copyright': 'Copyright (C) 2015-2019',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3+).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: quick and dirty timing facility
_EVENTS = []

# ======================================================================
# Greetings
MY_GREETINGS = r"""
 ____           _               ____                           _              
|  _ \ __ _ ___| |_ ___ _ __   / ___| ___  ___  _ __ ___   ___| |_ _ __ _   _ 
| |_) / _` / __| __/ _ \ '__| | |  _ / _ \/ _ \| '_ ` _ \ / _ \ __| '__| | | |
|  _ < (_| \__ \ ||  __/ |    | |_| |  __/ (_) | | | | | |  __/ |_| |  | |_| |
|_| \_\__,_|___/\__\___|_|     \____|\___|\___/|_| |_| |_|\___|\__|_|   \__, |
                                                                        |___/
"""
# generated with: figlet 'Raster Geometry' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
# print(MY_GREETINGS)

# ======================================================================
PATH = pkg_paths(__file__, INFO['name'], INFO['author'], INFO['version'])

# ======================================================================
elapsed(os.path.basename(__file__))

# ======================================================================
# : populate namespace
from raster_geometry.raster import (
    set_values, bresenham_line, bresenham_lines, bresenham_polygon,
    # bresenham_curve, bresenham_curves,
    center_of, as_vector, unit_vector, angle_between, signed_angle_2d,
    is_convex_2d, is_simple_2d, render_at, fill_convex, unfill,
    polygon, square, rectangle, rhombus, circle, ellipse,
    cube, cuboid, rhomboid, sphere, ellipsoid, cylinder,
    # polyhedron,
    extrema_to_semisizes_position,
    nd_lines,
    # nd_curves,
    nd_cuboid, nd_superellipsoid, nd_superellipsoidal_prism,
    # nd_cone, nd_superellipsoidal_cone,
    nd_gradient, nd_dirac_delta,
    # nd_polytope
    ellipsoid_specs, superellipsoid_specs, cuboid_specs, prism_specs,
    gradient_specs,
    multi_render,
)

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
