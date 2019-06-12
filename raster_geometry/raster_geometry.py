#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_geometry: create/manipulate N-dim raster geometric shapes.

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

The N-D geometrical shapes currently available are:
- cuboid: sum[abs(x_n/a_n)^inf] < 1
- superellipsoid: sum[abs(x_n/a_n)^k] < 1
- prism: stack (N-1)-D rendered objects on given axis

Running this file directly will run some tests.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import warnings  # Warning control
import random  # Generate pseudo-random numbers
import doctest  # Test interactive Python examples

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
from raster_geometry import INFO, PATH
from raster_geometry import elapsed, report
from raster_geometry import msg, dbg


# todo: support for Iterable relative position/size in nd_cuboid, etc.


# ======================================================================
def set_values(
        rendered,
        fill=(1, 0),
        dtype=float):
    """
    Set the values of a rendering pattern.

    Args:
        rendered (np.ndarray[bool]): The rendered object.
        fill (Sequence[Number|np.dtype]): Values to use.
            The first value is used for the rendered object.
            The second value is used for everything else.
        dtype (np.dtype): Desired output data-type.
            See `np.ndarray()` for more.

    Returns:
        arr (np.ndarray): The rendered object with custom values and data-type.
    """
    arr = np.empty_like(rendered, dtype=dtype)
    arr[rendered] = fill[0]
    arr[~rendered] = fill[1]
    return arr


# ======================================================================
def bresenham_line(
        coord_a,
        coord_b,
        endpoint=False):
    """
    Yield the integer points lying on an N-dim line.

    This uses an adaptation of the Bresenham algorithm.
    Note that this only works with integer coordinates.

    The size of the inputs must match.

    Args:
        coord_a (Sequence[int]): The coordinates of the starting point.
            The starting point is included.
        coord_b (Sequence[int]): The coordinates of the ending point.
            The endind point is excluded, unless `endpoint == True`.
        endpoint (bool): Determine wether to yield the last point.
            If True, the endpoint (`coord_b`) is yielded at last.
            Otherwise, the endpoint (`coord_b`) is not yielded.

    Yields:
        coord (tuple[int]): The coordinates of a point on the N-dim line.
    """
    n_dim = len(coord_a)
    diffs = [abs(b - a) for a, b in zip(coord_a, coord_b)]
    steps = [1 if a < b else -1 for a, b in zip(coord_a, coord_b)]
    max_diff = max(diffs)
    updates = [max_diff / 2] * n_dim
    coord = list(coord_a)
    for i in range(max_diff + 1):
        yield tuple(coord)
        for i, (x, d, s, u) in enumerate(
                zip(coord, diffs, steps, updates)):
            updates[i] -= d
            if u < 0:
                coord[i] += s
                updates[i] += max_diff
    if endpoint:
        yield tuple(coord_b)


# ======================================================================
def bresenham_lines(
        coords,
        closed=False):
    """
    Yield the integer points lying on N-dim lines.

    This uses an adaptation of the Bresenham algorithm.
    Note that this only works with integer coordinates.

    Args:
        coords (Sequence[Sequence[int]]): The coordinates of the points.
            The size of the items of `coords` must match.
        closed (bool): Render a line between the first and last points.
            This is used to generate closed lines.
            If only two points are given, this parameter has no effect.

    Yields:
        coord (tuple[int]): The coordinates of a point on the N-dim lines.

    See Also:
        bresenham_line()
    """
    for coord_a, coord_b in fc.util.slide(coords, 2):
        for coord in bresenham_line(coord_a, coord_b):
            yield coord
    if closed and len(coords) > 2:
        for coord in bresenham_line(coords[-1], coords[0]):
            yield coord


# ======================================================================
def bresenham_curve(
        coords,
        deg=2):
    """
    Yield the integer points lying on an N-dim Bézier curve of given degree.

    This uses an adaptation of the Bresenham algorithm.
    Note that this only works with integer coordinates.

    The size of the inputs must match.

    Args:
        coords (Sequence[int]): The coordinates of the Bézier points.
            The first and the last points are the start and end points.
            Intermediate points are used to compute the polynomial.
            The length of `coords` expected depends on the degree, as
            specified by `deg`, and must be equal to `deg + 1`.
        deg (int): The degree of the polynomial generating the curve.
            This must be larger than 0.
            If 1, this is equivalent to `bresenham_line()` but slower.

    Yields:
        coord (tuple[int]): The coordinates of a point on the N-dim curve.
    """
    # todo: implement generic brasenham_curve
    raise NotImplementedError


# ======================================================================
def bresenham_curves(
        coords,
        deg=2):
    """
    Yield the integer points lying on N-dim Bézier curves of given degree.

    This uses an adaptation of the Bresenham algorithm.
    Note that this only works with integer coordinates.

    All curves must have the same degree.


    Args:
        coords (Sequence[Sequence[int]]): The coordinates of the points.
            The size of the items of `coords` must match.
            The number of elements of `coords` are considered in groups of
            size `deg + 1` with
        deg (int): The degree of the polynomial generating the curve.
            This must be larger than 0.
            If 1, this is equivalent to `bresenham_lines()` but slower.

    Yields:
        coord (tuple[int]): The coordinates of a point on the N-dim curves.

    See Also:
        bresenham_curve(), bresenham_line(), bresenham_lines()
    """
    gen_coords = [coords[i:len(coords) - deg + i:deg] for i in range(deg + 1)]
    for _coords in zip(*gen_coords):
        for coord in bresenham_curve(_coords):
            yield coord


# ======================================================================
def center_of(
        coords,
        as_int=True):
    """
    Compute the geometric center of a number of points.

    Args:
        coords (Sequence[Sequence[int|float]): The input points.
            These are the coordinates of the points.
        as_int (bool): Approximate the result as integer.

    Returns:
        center (tuple[int|float]): The geometric center coordinates.
            The result is generally a fractional number.
            If `as_int` is True, this is approximated to the closest integer.
    """
    center = coords[0]
    for i, coord in enumerate(coords[1:]):
        center = tuple(
            ((i + 1) * x0 + x) / (i + 2) for x, x0 in zip(coord, center))
    if as_int:
        center = tuple(int(round(x)) for x in center)
    return center


# ======================================================================
def as_vector(
        coord,
        origin=None):
    """
    Interpret the point as a vector.

    This is useful to perform mathematical operations using the standard
    syntax.

    Args:
        coord (Sequence[int|float]): The N-dim point.
        origin (Sequence[int|float]): An N-dim point used as origin.
            If None, the null vector is used as the origin.

    Returns:
        vector (np.ndarray): The vector with respect to the origin.

    Examples:
        >>> as_vector([0, 3, 4])
        array([0, 3, 4])
        >>> as_vector([5, 0, 0])
        array([5, 0, 0])
    """
    origin = np.array(origin) if origin else 0
    return np.array(coord) - origin


# ======================================================================
def unit_vector(vector):
    """
    Compute the unit vector.

    Args:
        vector (np.ndarray): The input vector.

    Returns:
        u_vector (np.ndarray): The corresponding unit vector.

    Examples:
        >>> print(unit_vector(as_vector([0, 3, 4])))
        [0.  0.6 0.8]
        >>> print(unit_vector(as_vector([5, 0, 0])))
        [1. 0. 0.]
    """
    return vector / np.linalg.norm(vector)


# ======================================================================
def angle_between(
        coord_a,
        coord_b,
        origin=None):
    """
    Compute the angle between two vectors.

    The vectors are identified by the given coordinates with respect to the
    specified origin.

    Args:
        coord_a (Sequence[int|float]): The first input coordinates.
        coord_b (Sequence[int|float]): The second input coordinates.
        origin (Sequence[int|float]|None): The coordinates of the origin.

    Returns:
        angle (float): The angle between the two vectors in rad.

    Examples:
        >>> np.rad2deg(angle_between([0, 3, 4], [5, 0, 0]))
        90.0
        >>> angle = np.rad2deg(angle_between([0, 3, 4], [5, 0, 0], [1, 1, 1]))
        >>> print(np.round(angle, 2))
        124.54
    """
    u_vector_a = unit_vector(as_vector(coord_a, origin))
    u_vector_b = unit_vector(as_vector(coord_b, origin))
    angle = np.arccos(np.clip(np.dot(u_vector_a, u_vector_b), -1.0, 1.0))
    return angle


# ======================================================================
def signed_angle_2d(
        coord,
        origin=None):
    """
    Compute the (signed) angle of a vector with respect to the first axis.

    This only works in 2D.

    Args:
        coord (Sequence[int|float]): The input coordinates.
        origin (Sequence[int|float]|None): The coordinates of the origin.

    Returns:
        angle (float): The angle between the two vectors in rad.

    Examples:
        >>> np.rad2deg(signed_angle_2d([1, 0]))
        0.0
        >>> np.rad2deg(signed_angle_2d([0, 1]))
        90.0
        >>> np.rad2deg(signed_angle_2d([1, 1]))
        45.0
        >>> np.rad2deg(signed_angle_2d([1, 1, 4]))
        Traceback (most recent call last):
            ...
        AssertionError
    """
    u_vector = unit_vector(as_vector(coord, origin))
    assert (u_vector.shape == (2,))
    return np.arctan2(u_vector[1], u_vector[0])


# ======================================================================
def is_convex_2d(coords):
    """
    Determine if a simple polygon is convex.

    This only works for simple polygons.

    Args:
        coords (Sequence[Sequence[int|float]]): The vertices of the polygon.

    Returns:
        is_convex (bool): If the specified polygon is convex.
    """
    coords = list(coords) + list(coords[:2])
    dets = [
        np.cross(as_vector(point_b, point_a), as_vector(point_c, point_b))
        for point_a, point_b, point_c in
        zip(coords[:-2], coords[1:-1], coords[2:])]
    return all([det > 0 for det in dets]) or all([det < 0 for det in dets])


# ======================================================================
def is_simple_2d(coords):
    """
    Determine if a simple polygon is convex.

    This only works for simple polygons.

    Args:
        coords (Sequence[Sequence[int|float]]): The vertices of the polygon.

    Returns:
        is_convex (bool): If the specified polygon is convex.
    """
    coords = list(coords) + list(coords[:2])
    dets = [
        np.cross(as_vector(point_b, point_a), as_vector(point_c, point_b))
        for point_a, point_b, point_c in
        zip(coords[:-2], coords[1:-1], coords[2:])]
    return all([det > 0 for det in dets]) or all([det < 0 for det in dets])


# ======================================================================
def render_at(
        shape,
        coords):
    """
    Render at specific integer coordinates in a shape.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        coords (Sequence[tuple(int)]): The coordinates to render.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    rendered = np.zeros(shape, dtype=bool)
    # : slower
    # mask = tuple(zip(*coords))
    # rendered[mask] = True
    for point in coords:
        rendered[point] = True
    return rendered


# ======================================================================
def fill_convex(
        arr,
        border=True,
        fill=True):
    """
    Fill a convex N-dimensional geometrical object from its outer shape.

    Args:
        arr (np.ndarray): The array containing the outer shape of the object.
        border: The value to use to identify the border.
        fill: The value to use for the filling.

    Returns:
        arr (np.ndarray): The filled geometrical object.

    Examples:
        >>> points = ((1, 1), (0, 6), (5, 4), (4, 2))
        >>> line_points = tuple(x for x in bresenham_lines(points, True))
        >>> arr = render_at((8, 8), line_points)
        >>> print(arr)
        [[False False False False  True  True  True False]
         [False  True  True  True False False  True False]
         [False  True False False False  True False False]
         [False False  True False False  True False False]
         [False False  True False  True False False False]
         [False False False  True  True False False False]
         [False False False False False False False False]
         [False False False False False False False False]]
        >>> arr = fill_convex(arr)
        >>> print(arr)
        [[False False False False  True  True  True False]
         [False  True  True  True  True  True  True False]
         [False  True  True  True  True  True False False]
         [False False  True  True  True  True False False]
         [False False  True  True  True False False False]
         [False False False  True  True False False False]
         [False False False False False False False False]
         [False False False False False False False False]]
    """
    shape = arr.shape
    arr = arr.reshape((shape[0], -1))
    for i in range(arr.shape[1]):
        mask = slice(None), slice(i, i + 1)
        line = arr[mask]
        j = np.where(line == border)[0]
        if len(j):
            start, stop = np.min(j), np.max(j)
            line[start:stop] = fill
    return arr.reshape(shape)


# ======================================================================
def unfill(
        arr,
        structure=None,
        border=1):
    """
    Fill a convex N-dimensional geometrical object from its outer shape.

    Args:
        arr (np.ndarray): The array containing the outer shape of the object.
        structure: The structuring element for the erosion.
            See `structure` parameter of `scipy.ndimage.binary_erosion()`.
        border (int): The border thickness in px.
            See `iterations` parameter of `scipy.ndimage.binary_erosion()`.

    Returns:
        arr (np.ndarray): The filled geometrical object.

    Examples:
        >>> points = ((1, 1), (0, 6), (5, 4), (4, 2))
        >>> line_points = tuple(x for x in bresenham_lines(points, True))
        >>> arr = render_at((8, 8), line_points)
        >>> print(arr)
        [[False False False False  True  True  True False]
         [False  True  True  True False False  True False]
         [False  True False False False  True False False]
         [False False  True False False  True False False]
         [False False  True False  True False False False]
         [False False False  True  True False False False]
         [False False False False False False False False]
         [False False False False False False False False]]
        >>> np.all(unfill(arr) == arr)
        True
        >>> full_arr = fill_convex(arr)
        >>> print(full_arr)
        [[False False False False  True  True  True False]
         [False  True  True  True  True  True  True False]
         [False  True  True  True  True  True False False]
         [False False  True  True  True  True False False]
         [False False  True  True  True False False False]
         [False False False  True  True False False False]
         [False False False False False False False False]
         [False False False False False False False False]]
        >>> hollow_arr = unfill(full_arr.T).T
        >>> print(hollow_arr)
        [[False False False False  True  True  True False]
         [False  True  True  True False False  True False]
         [False  True False False False  True False False]
         [False False  True False False  True False False]
         [False False  True False  True False False False]
         [False False False  True  True False False False]
         [False False False False False False False False]
         [False False False False False False False False]]
    """
    eroded = sp.ndimage.binary_erosion(arr, structure, border)
    return arr ^ eroded


# ======================================================================
def polygon(
        shape,
        positions,
        sorting=True,
        filling=False,
        rel_position=True):
    """
    Render a simple polygon.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        positions (Sequence[float|Sequence[float]]): The position of vertices.
            The first Sequence must consist of three or more items,
            each describing a point as an Sequence of floats
            (with length matching the desired number of dimensions)
            or a single float (repeated in all dimensions).
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        sorting (bool): Sort the positions counter-clockwise.
            This ensures that the resulting polygon is simple, i.e.
            it is not self-intersecting.
            Note that if the positions describe a non-convex polygon,
            a different order of the positions of the vertices could result
            in a different simple polygon.
            For convex polygons, such ambiguity does not exists.
        filling (bool): Fill the points inside the polygon.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `flyingcircus.extra.grid()` internally.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(polygon((9, 9), ((0.2, 0.2), (0.2, 0.8), (0.7, 0.5))))
        [[False False False False False False False False False]
         [False  True  True  True  True  True  True  True False]
         [False False  True False False False  True False False]
         [False False  True False False False  True False False]
         [False False False  True False  True False False False]
         [False False False  True False  True False False False]
         [False False False False  True False False False False]
         [False False False False False False False False False]
         [False False False False False False False False False]]
        >>> print(polygon(
        ...    (9, 9), ((0.2, 0.2), (0.2, 0.8), (0.7, 0.5)), filling=True))
    """
    n_dim = 2
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    positions = [
        fc.util.auto_repeat(position, n_dim, check=True)
        for position in positions]
    coords = [
        fc.num.coord(shape, position, rel_position, True)
        for position in positions]
    # : sort vertices
    if sorting:
        center = center_of(coords)
        angles = [signed_angle_2d(coord, center) for coord in coords]
        coords = list(list(zip(*sorted(zip(angles, coords))))[1])
    # : render polygon
    points = bresenham_lines(coords, True)
    if filling:
        print(set(points))
        points = bresenham_lines(list(points), True)
        print(points)
    return render_at(shape, set(points))


# ======================================================================
def square(
        shape,
        side,
        position=0.5):
    """
    Render a square.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        side (float): The side of the square in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(square(4, 2))
        [[False False False False]
         [False  True  True False]
         [False  True  True False]
         [False False False False]]
        >>> print(square(5, 3))
        [[False False False False False]
         [False  True  True  True False]
         [False  True  True  True False]
         [False  True  True  True False]
         [False False False False False]]
    """
    return nd_cuboid(
        shape, side / 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rectangle(
        shape,
        semisides,
        position=0.5):
    """
    Render a rectangle.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semisides (Sequence[float]): The semisides of the rectangle in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(rectangle(6, (2, 1)))
        [[False False False False False False]
         [False False  True  True False False]
         [False False  True  True False False]
         [False False  True  True False False]
         [False False  True  True False False]
         [False False False False False False]]
        >>> print(rectangle(5, (2, 1)))
        [[False  True  True  True False]
         [False  True  True  True False]
         [False  True  True  True False]
         [False  True  True  True False]
         [False  True  True  True False]]
        >>> print(rectangle(4, (1, 0.5), 0))
        [[ True False False False]
         [ True False False False]
         [False False False False]
         [False False False False]]
        >>> print(rectangle(4, (2, 1), 0))
        [[ True  True False False]
         [ True  True False False]
         [ True  True False False]
         [False False False False]]
    """
    return nd_cuboid(
        shape, semisides, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rhombus(
        shape,
        semidiagonals,
        position=0.5):
    """
    Render a rhombus.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semidiagonals (float|Sequence[float]): The rhombus semidiagonas in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(rhombus(5, 2))
        [[False False  True False False]
         [False  True  True  True False]
         [ True  True  True  True  True]
         [False  True  True  True False]
         [False False  True False False]]
        >>> print(rhombus(5, (2, 1)))
        [[False False  True False False]
         [False False  True False False]
         [False  True  True  True False]
         [False False  True False False]
         [False False  True False False]]
    """
    return nd_superellipsoid(
        shape, semidiagonals, 1.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def circle(
        shape,
        radius,
        position=0.5):
    """
    Render a circle.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        radius (float): The radius of the circle in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(circle(5, 1))
        [[False False False False False]
         [False False  True False False]
         [False  True  True  True False]
         [False False  True False False]
         [False False False False False]]
        >>> print(circle(6, 2))
        [[False False False False False False]
         [False False  True  True False False]
         [False  True  True  True  True False]
         [False  True  True  True  True False]
         [False False  True  True False False]
         [False False False False False False]]
        >>> print(circle(4, 2, 0))
        [[ True  True  True False]
         [ True  True False False]
         [ True False False False]
         [False False False False]]
    """
    return nd_superellipsoid(
        shape, radius, 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def ellipse(
        shape,
        semiaxes,
        position=0.5):
    """
    Render an ellipse.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semiaxes (float|Sequence[float]): The semiaxes of the ellipse in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(ellipse(6, (2, 3)))
        [[False False False False False False]
         [False  True  True  True  True False]
         [ True  True  True  True  True  True]
         [ True  True  True  True  True  True]
         [False  True  True  True  True False]
         [False False False False False False]]
        >>> print(ellipse(6, (5, 3), 0))
        [[ True  True  True  True False False]
         [ True  True  True False False False]
         [ True  True  True False False False]
         [ True  True  True False False False]
         [ True  True False False False False]
         [ True False False False False False]]
    """
    return nd_superellipsoid(
        shape, semiaxes, 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def cube(
        shape,
        side,
        position=0.5):
    """
    Render a cube.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        side (float): The side of the cube in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(cube(4, 2))
        [[[False False False False]
          [False False False False]
          [False False False False]
          [False False False False]]
        <BLANKLINE>
         [[False False False False]
          [False  True  True False]
          [False  True  True False]
          [False False False False]]
        <BLANKLINE>
         [[False False False False]
          [False  True  True False]
          [False  True  True False]
          [False False False False]]
        <BLANKLINE>
         [[False False False False]
          [False False False False]
          [False False False False]
          [False False False False]]]
    """
    return nd_cuboid(
        shape, side / 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def cuboid(
        shape,
        semisides,
        position=0.5):
    """
    Render a cuboid.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semisides (Sequence[float]): The semisides of the cuboid in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(cuboid((3, 4, 6), (0.5, 2, 1)))
        [[[False False False False False False]
          [False False False False False False]
          [False False False False False False]
          [False False False False False False]]
        <BLANKLINE>
         [[False False  True  True False False]
          [False False  True  True False False]
          [False False  True  True False False]
          [False False  True  True False False]]
        <BLANKLINE>
         [[False False False False False False]
          [False False False False False False]
          [False False False False False False]
          [False False False False False False]]]
    """
    return nd_cuboid(
        shape, semisides, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rhomboid(
        shape,
        semidiagonals,
        position=0.5):
    """
    Render a rhomboid.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semidiagonals (Sequence[float]): The rhomboid semidiagonals in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(rhomboid((3, 5, 7), (1, 1, 2)))
        [[[False False False False False False False]
          [False False False False False False False]
          [False False False  True False False False]
          [False False False False False False False]
          [False False False False False False False]]
        <BLANKLINE>
         [[False False False False False False False]
          [False False False  True False False False]
          [False  True  True  True  True  True False]
          [False False False  True False False False]
          [False False False False False False False]]
        <BLANKLINE>
         [[False False False False False False False]
          [False False False False False False False]
          [False False False  True False False False]
          [False False False False False False False]
          [False False False False False False False]]]
    """
    return nd_superellipsoid(
        shape, semidiagonals, 1.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def sphere(
        shape,
        radius,
        position=0.5):
    """
    Render a sphere.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        radius (float): The radius of the sphere in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(sphere(3, 1))
        [[[False False False]
          [False  True False]
          [False False False]]
        <BLANKLINE>
         [[False  True False]
          [ True  True  True]
          [False  True False]]
        <BLANKLINE>
         [[False False False]
          [False  True False]
          [False False False]]]
        >>> print(sphere(5, 2))
        [[[False False False False False]
          [False False False False False]
          [False False  True False False]
          [False False False False False]
          [False False False False False]]
        <BLANKLINE>
         [[False False False False False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False False False False False]]
        <BLANKLINE>
         [[False False  True False False]
          [False  True  True  True False]
          [ True  True  True  True  True]
          [False  True  True  True False]
          [False False  True False False]]
        <BLANKLINE>
         [[False False False False False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False False False False False]]
        <BLANKLINE>
         [[False False False False False]
          [False False False False False]
          [False False  True False False]
          [False False False False False]
          [False False False False False]]]
    """
    return nd_superellipsoid(
        shape, radius, 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def ellipsoid(
        shape,
        semiaxes,
        position=0.5):
    """
    Render an ellipsoid.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semiaxes (float|Sequence[float]): The semiaxes of the ellipsoid in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(ellipsoid(5, (1., 2., 1.5)))
        [[[False False False False False]
          [False False False False False]
          [False False False False False]
          [False False False False False]
          [False False False False False]]
        <BLANKLINE>
         [[False False False False False]
          [False False False False False]
          [False False  True False False]
          [False False False False False]
          [False False False False False]]
        <BLANKLINE>
         [[False False  True False False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False  True  True  True False]
          [False False  True False False]]
        <BLANKLINE>
         [[False False False False False]
          [False False False False False]
          [False False  True False False]
          [False False False False False]
          [False False False False False]]
        <BLANKLINE>
         [[False False False False False]
          [False False False False False]
          [False False False False False]
          [False False False False False]
          [False False False False False]]]
    """
    return nd_superellipsoid(
        shape, semiaxes, 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def cylinder(
        shape,
        height,
        radius,
        axis=-1,
        position=0.5):
    """
    Render a cylinder.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        height (float): The height of the cylinder in px.
        radius (float): The radius of the cylinder in px.
        axis (int): Orientation of the cylinder in the N-dim space.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(cylinder(4, 2, 2, 0))
        [[[False False False False]
          [False False False False]
          [False False False False]
          [False False False False]]
        <BLANKLINE>
         [[False  True  True False]
          [ True  True  True  True]
          [ True  True  True  True]
          [False  True  True False]]
        <BLANKLINE>
         [[False  True  True False]
          [ True  True  True  True]
          [ True  True  True  True]
          [False  True  True False]]
        <BLANKLINE>
         [[False False False False]
          [False False False False]
          [False False False False]
          [False False False False]]]
    """
    n_dim = 3
    shape = fc.util.auto_repeat(shape, n_dim)
    position = fc.util.auto_repeat(position, n_dim)
    # generate base
    base_shape = tuple(
        dim for i, dim in enumerate(shape) if axis % n_dim != i)
    base_position = tuple(
        dim for i, dim in enumerate(position) if axis % n_dim != i)
    base = circle(base_shape, radius, base_position)
    # use N-dim function
    return nd_prism(
        base, shape[axis], axis, height, position[axis],
        rel_position=True, rel_sizes=False)


# ======================================================================
def extrema_to_semisizes_position(minima, maxima, num=None):
    """
    Compute semisizes and positions from extrema.

    Args:
        minima (float|Sequence[float]): The minimum extrema.
            These are the lower bound of the object.
            If both `minima` and `maxima` are Sequence, their length must
            match.
        maxima (float|Sequence[float]): The maximum extrema.
            These are the upper bound of the object.
            If both `minima` and `maxima` are Sequence, their length must
            match.
        num (int|None): The number of extrema.
            If None, it is guessed from the `minima` and/or `maxima`.
            If int, must match the length of `minima` and/or `maxima`
            (if they are Sequence).

    Returns:
        result (tuple): The tuple
            contains:
             - semisizes (Sequence[float]): The N-dim cuboid semisides sizes.
             - position (Sequence[float]): The position of the center.
               Values are relative to the lowest edge.

    Examples:
        >>> extrema_to_semisizes_position(0.1, 0.5)
        ([0.2], [0.3])
        >>> extrema_to_semisizes_position(0.2, 0.5)
        ([0.15], [0.35])
        >>> extrema_to_semisizes_position(0.2, 0.5, 2)
        ([0.15, 0.15], [0.35, 0.35])
        >>> extrema_to_semisizes_position((0.2, 0.1), 0.5)
        ([0.15, 0.2], [0.35, 0.3])
        >>> extrema_to_semisizes_position((0.1, 0.2), (0.9, 0.5))
        ([0.4, 0.15], [0.5, 0.35])
    """
    if not num:
        num = fc.util.combine_iter_len((minima, maxima, num))
    # check compatibility of given parameters
    minima = fc.util.auto_repeat(minima, num, check=True)
    maxima = fc.util.auto_repeat(maxima, num, check=True)
    semisizes, position = [], []
    for min_val, max_val in zip(minima, maxima):
        semisizes.append((max_val - min_val) / 2.0)
        position.append((max_val + min_val) / 2.0)
    return semisizes, position


# ======================================================================
def polyhedron(
        shape,
        vertices,
        position=0.5):
    """
    Render a simple polyhedron.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        vertices (Sequence[Sequence[float]): Coordinates of the vertices in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    raise NotImplementedError


# ======================================================================
def nd_lines(
        shape,
        positions,
        closed=False,
        n_dim=None,
        rel_position=True):
    """
    Render a series of N-dim lines.

    The positions are aligned to the integer grid defined by the shape.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        positions (Sequence[float|Sequence[float]]): The position of extrema.
            The first Sequence must consist of two or more items,
            each describing a point as an Sequence of floats
            (with length matching the desired number of dimensions)
            or a single float (repeated in all dimensions).
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        closed (bool): Render a line between the first and last positions.
            This is used to generate closed lines.
            If only two positions are given, this parameter has no effect.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `position` must be a sequence.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `flyingcircus.extra.coord()` internally.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    See Also:
        bresenham_line(), bresenham_lines()
    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape,) + tuple(positions))
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    positions = [
        fc.util.auto_repeat(position, n_dim, check=True)
        for position in positions]
    coords = [
        fc.num.coord(shape, position, rel_position, True)
        for position in positions]
    rendered = render_at(shape, bresenham_lines(coords, closed))
    return rendered


# ======================================================================
def nd_curves(
        shape,
        positions,
        deg=2,
        n_dim=None,
        rel_position=True):
    """
    Render a series of N-dim Bézier curves of given degree.

    The positions are aligned to the integer grid defined by the shape.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        positions (Sequence[float|Sequence[float]]): The position of extrema.
            The first Sequence must consist of two or more items,
            each describing a point as an Sequence of floats
            (with length matching the desired number of dimensions)
            or a single float (repeated in all dimensions).
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        deg (int): The degree of the curves.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `positions` must be a sequence.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `flyingcircus.extra.coord()` internally.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    # todo: handle curve-degree properly
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape,) + tuple(positions))
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    positions = [
        fc.util.auto_repeat(position, n_dim, check=True)
        for position in positions]
    coords = [
        fc.num.coord(shape, position, rel_position, True)
        for position in positions]
    rendered = render_at(shape, bresenham_curves(coords, deg))
    return rendered


# ======================================================================
def nd_cuboid(
        shape,
        semisizes=0.5,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Render an N-dim cuboid (hypercuboid).

    The cartesian equations are:

    .. math::
        \\sum[\\abs(\\frac{x_n}{a_n})^{\\inf}] < 1.0

    where :math:`n` runs through the dims, :math:`x` are the cartesian
    coordinate, :math:`a` are the semi-sizes (semi-axes) and
    :math:`\\inf` is infinity.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semisizes (float|Sequence[float]): The cuboid semisides sizes.
            The values interpretation depend on `rel_sizes`.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `position`, `semisizes` must be a sequence.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `position` using `shape`.
            Uses `flyingcircus.extra.grid_coord()` internally, see its
            `is_relative`
            parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `semisizes` using `shape`.
            Uses `flyingcircus.extra.coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape, position, semisizes))
    # check compatibility of given parameters
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    position = fc.util.auto_repeat(position, n_dim, check=True)
    semisizes = fc.util.auto_repeat(semisizes, n_dim, check=True)
    # fix relative units
    semisizes = fc.num.coord(
        shape, semisizes, is_relative=rel_sizes, use_int=False)
    xx = fc.num.grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    # create the rendered object
    rendered = np.ones(shape, dtype=bool)
    for x_i, semisize in zip(xx, semisizes):
        rendered *= (np.abs(x_i) <= semisize)
    return rendered


# ======================================================================
def nd_superellipsoid(
        shape,
        semisizes=0.5,
        indexes=2,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Render an N-dim superellipsoid.

    The cartesian equations are:

    .. math::
        \\sum[\\abs(\\frac{x_n}{a_n})^{k_n}] < 1.0

    where :math:`n` runs through the dims, :math:`x` are the cartesian
    coordinate, :math:`a` are the semi-sizes (semi-axes) and
    :math:`k` are the indexes.

    When the index is 2, an ellipsoid (hyperellipsoid) is generated.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        semisizes (float|Sequence[float]): The superellipsoid axes semisizes.
            The values interpretation depend on `rel_sizes`.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        indexes (float|Sequence[float]): The exponent of the summed terms.
            If 2, generates ellipsoids.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `position`, `semisizes`, `indexes` must be
            Sequence.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `position`.
            Uses `flyingcircus.extra.grid_coord()` internally, see its
            `is_relative` parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `semisizes`.
            Uses `flyingcircus.extra.coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape, position, semisizes, indexes))

    # check compatibility of given parameters
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    position = fc.util.auto_repeat(position, n_dim, check=True)
    semisizes = fc.util.auto_repeat(semisizes, n_dim, check=True)
    indexes = fc.util.auto_repeat(indexes, n_dim, check=True)

    # get correct position
    semisizes = fc.num.coord(
        shape, semisizes, is_relative=rel_sizes, use_int=False)
    # print('Semisizes: {}'.format(semisizes))  # DEBUG
    # print('Shape: {}'.format(shape))  # DEBUG
    xx = fc.num.grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    # print('X: {}'.format(xx))  # DEBUG

    # create the rendered
    rendered = np.zeros(shape, dtype=float)
    for x_i, semisize, index in zip(xx, semisizes, indexes):
        rendered += (np.abs(x_i / semisize) ** index)
    rendered = rendered <= 1.0

    return rendered


# ======================================================================
def nd_prism(
        base,
        extra_dim,
        axis=-1,
        size=0.5,
        position=0.5,
        rel_position=True,
        rel_sizes=True):
    """
    Render a N-dim prism.

    Args:
        base (np.ndarray): Base (N-1)-dim rendered object to stack as prism.
        extra_dim (int): Size of the new dimension to be added.
        axis (int): Orientation of the prism in the N-dim space.
        size (float): The size of the prism height.
            The values interpretation depend on `rel_size`.
        position (float): The relative position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
            This setting only affects the extra shape dimension.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `position` using `extra_dim`.
            Uses `flyingcircus.extra.grid_coord()` internally, see its
            `is_relative`
            parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `size` using `extra_dim`.
            Uses `flyingcircus.extra.coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    n_dim = base.ndim + 1
    if axis > n_dim:
        raise ValueError(
            'axis of orientation must not exceed the number of dimensions')
    # get correct position
    size = fc.num.coord(
        (extra_dim,), size, is_relative=rel_sizes, use_int=False)[0]
    xx = fc.num.grid_coord(
        (extra_dim,), (position,), is_relative=rel_position, use_int=False)[0]
    extra_rendered = np.abs(xx) <= (size / 2.0)
    # calculate rendered object shape
    shape = (
            base.shape[:axis] + (extra_dim,) + base.shape[axis:])
    # create indefinite prism
    rendered = np.zeros(shape, dtype=bool)
    for i in range(extra_dim):
        if extra_rendered[i]:
            index = [slice(None)] * n_dim
            index[axis] = i
            rendered[tuple(index)] = base
    return rendered


# ======================================================================
def nd_superellipsoidal_prism(
        shape,
        axis=-1,
        semisizes=0.5,
        indexes=2,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """

    Args:
        shape:
        axis:
        semisizes:
        indexes:
        position:
        n_dim:
        rel_position:
        rel_sizes:

    Returns:

    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape, position, semisizes))
    axis = axis % n_dim
    # separate shape/dims
    base_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)
    extra_dim = shape[axis]
    # separate position
    position = fc.util.auto_repeat(position, n_dim, False, True)
    base_position = tuple(x for i, x in enumerate(position) if i != axis)
    extra_position = position[axis]
    # separate semisizes
    semisizes = fc.util.auto_repeat(semisizes, n_dim, False, True)
    base_semisizes = tuple(x for i, x in enumerate(semisizes) if i != axis)
    extra_semisize = semisizes[axis]
    # generate prism base
    base = nd_superellipsoid(
        base_shape, base_semisizes, indexes, base_position, n_dim - 1,
        rel_position, rel_sizes)
    # generate final prism
    rendered = nd_prism(
        base, extra_dim, axis, extra_semisize * 2, extra_position,
        rel_position, rel_sizes)
    return rendered


# ======================================================================
def nd_cone(
        base,
        extra_dim,
        axis=-1,
        size=0.5,
        position=0.5,
        rel_position=True,
        rel_sizes=True):
    raise NotImplementedError


# ======================================================================
def nd_superellipsoidal_cone(
        shape,
        semisizes=0.5,
        indexes=2,
        axis=-1,
        size=0.5,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    raise NotImplementedError


# ======================================================================
def nd_gradient(
        shape,
        gen_ranges=None,
        dtype=float,
        dense=False,
        generators=np.linspace,
        generators_kws=None,
        n_dim=None):
    """
    Generate N-dimensional gradient data arrays.

    This is useful for generating simulation data patterns.

    Args:
        shape (int|Sequence[int]|None): The shape of the container in px.
            If None, this is determined from the slice.step in `gen_ranges`.
            At least one of `shape` and the slice.step info in `gen_ranges`
            must be specified.
            If both are specified, the info therein must match.
        gen_ranges (Sequence[tuple]|None): Generator ranges and numbers.
            An Sequence of 2 or 3-tuples: the first two are always interpreted
            as the the extrema of the range to cover, while the third (when
            present) indicates the number of samples.
            If the number of samples is not specified (or is set to None),
            this info is taken from the `shape` parameter.
            If both `shape` and the third tuple element are specified,
            the info therein must match.
            If None, the shape information is used to generate the values,
            using (0, dim - 1, dim) for dim in shape.
        dtype (data-type): Desired output data-type.
            See `np.ndarray()` for more.
            If `gen_ranges` is None, defaults to `int`.
        dense (bool): Generate dense results.
            If True, the results have full sizes.
            Otherwise, constant dimensions are keept to size of 1,
            which ensures them being broadcast-safe.
        generators (callable|Sequence[callable]): The range generator(s).
            A generator `func` must have signature:
            func(any, any, int, **_kws) -> Sequence
            If callable, the same generator is used for all dimensions.
            If Sequence, each generator generate the data corresponding to its
            position in the Sequence, and its length must match the length of
            `gen_ranges`.
        generators_kws (dict|Sequence[dict|None]|None): Keyword arguments.
            If None, no arguments are passed to the generator.
            If dict, the same object is passed to all instances of
            `generators`.
            If Sequence, each dict (or None) is applied to the corresponding
            generator in `generators`, and its length must match the length of
            `generators`.


    Returns:
        arrs (list[np.ndarray]): The broadcast-safe N-dim gradient arrays.
            The actual shape depend on `dense`.

    Examples:
        >>> for arr in nd_gradient(None, ((0, 1, 2), (-2, 2, 2))):
        ...     print(arr)
        [[0.]
         [1.]]
        [[-2.  2.]]
        >>> for arr in nd_gradient(
        ...         None, ((0, 1, 2), (-2, 2, 2)), dense=True):
        ...     print(arr)
        [[0. 0.]
         [1. 1.]]
        [[-2.  2.]
         [-2.  2.]]
        >>> for arr in nd_gradient(
        ...         None, ((0, 1, 2), (-2, 2, 2)), int, True):
        ...     print(arr)
        [[0 0]
         [1 1]]
        [[-2  2]
         [-2  2]]
        >>> for arr in nd_gradient(
        ...         None, ((0, 1, 2), (-2, 2, 2)), float, True,
        ...         np.logspace):
        ...     print(arr)
        [[ 1.  1.]
         [10. 10.]]
        [[1.e-02 1.e+02]
         [1.e-02 1.e+02]]
        >>> for arr in nd_gradient(
        ...         None, ((0, 1, 2), (-2, 2, 2)), float, True,
        ...         (np.linspace, np.logspace)):
        ...     print(arr)
        [[0. 0.]
         [1. 1.]]
        [[1.e-02 1.e+02]
         [1.e-02 1.e+02]]

        >>> for arr in nd_gradient(
        ...         None, ((0, 1, 2), (-1, 1, 3), (-2, 2, 2)),
        ...         int, True):
        ...     print(arr)
        [[[0 0]
          [0 0]
          [0 0]]
        <BLANKLINE>
         [[1 1]
          [1 1]
          [1 1]]]
        [[[-1 -1]
          [ 0  0]
          [ 1  1]]
        <BLANKLINE>
         [[-1 -1]
          [ 0  0]
          [ 1  1]]]
        [[[-2  2]
          [-2  2]
          [-2  2]]
        <BLANKLINE>
         [[-2  2]
          [-2  2]
          [-2  2]]]
        >>> for arr in nd_gradient((2, 20)):
        ...     print(arr)
        [[0]
         [1]]
        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]]
        >>> for arr in nd_gradient((3, 16), dense=True):
        ...     print(arr)
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
         [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]]
        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
         [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
         [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]]
        >>> for arr in nd_gradient((3, 5), (4, 6), dense=True):
        ...     print(arr)
        [[4. 4. 4. 4. 4.]
         [5. 5. 5. 5. 5.]
         [6. 6. 6. 6. 6.]]
        [[4.  4.5 5.  5.5 6. ]
         [4.  4.5 5.  5.5 6. ]
         [4.  4.5 5.  5.5 6. ]]
        >>> for arr in nd_gradient((3, 5), ((4, 6), (3, 5))):
        ...     print(arr)
        [[4.]
         [5.]
         [6.]]
        [[3.  3.5 4.  4.5 5. ]]
        >>> for arr in nd_gradient((3, 5), ((4, 6, 3), (3, 5))):
        ...     print(arr)
        [[4.]
         [5.]
         [6.]]
        [[3.  3.5 4.  4.5 5. ]]
        >>> for arr in nd_gradient((3, 5), ((4, 6), (3, 5, 5))):
        ...     print(arr)
        [[4.]
         [5.]
         [6.]]
        [[3.  3.5 4.  4.5 5. ]]
        >>> for arr in nd_gradient((3, 5), ((4, 6), (3, 5, 5))):
        ...     print(arr)
        [[4.]
         [5.]
         [6.]]
        [[3.  3.5 4.  4.5 5. ]]
        >>> nd_gradient((3, 5), ((4, 6, 5), (4, 6, 5)))
        Traceback (most recent call last):
            ...
        ValueError: Info in `shape` and `gen_shapes` do not match.
        >>> nd_gradient((3, 5), ((4, 6, None),))
        Traceback (most recent call last):
            ...
        ValueError: Incompatible input value length.
    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape, gen_ranges))

    generators = fc.util.auto_repeat(
        generators, n_dim, check=True)
    generators_kws = fc.util.auto_repeat(
        generators_kws, n_dim, check=True)

    if not shape:
        shape = tuple(gen_range[2] for gen_range in gen_ranges)

    if not gen_ranges:
        gen_ranges = tuple((0, dim - 1, dim) for dim in shape)
        dtype = int
    else:
        nesting_level = fc.util.nesting_level(gen_ranges)
        gen_ranges = fc.util.auto_repeat(
            gen_ranges, n_dim, nesting_level == 1, True)

    arrs = []
    for i, (dim, gen_range, generator, generator_kws) in \
            enumerate(zip(shape, gen_ranges, generators, generators_kws)):
        if len(gen_range) == 2:
            num = dim
            start, stop = gen_range
        else:
            start, stop, num = gen_range
        if not num:
            num = dim
        elif num != dim:
            raise ValueError('Info in `shape` and `gen_shapes` do not match.')
        if generator_kws is None:
            generator_kws = {}
        arr = np.array(
            generator(start, stop, num, **generator_kws), dtype=dtype)
        new_shape = tuple(n if j == i else 1 for j, n in enumerate(shape))
        arr = arr.reshape(new_shape)
        if dense:
            arr = np.zeros(shape, dtype=dtype) + arr
        arrs.append(arr)
    return arrs


# ======================================================================
def nd_dirac_delta(
        shape,
        position=0.5,
        value=np.inf,
        n_dim=None,
        rel_position=True):
    """
    Generate an approximation of Dirac's Delta function.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        value (int|float): The value of the peak.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `position` must be a sequence.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `position` using `shape`.
            Uses `fc.num.grid_coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> print(nd_dirac_delta((5, 5), 0.5, 1))
        [[0 0 0 0 0]
         [0 0 0 0 0]
         [0 0 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 0]]
        >>> print(nd_dirac_delta(4, (0.5, 0.5), 9.0))
        [[0. 0. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 9. 0.]
         [0. 0. 0. 0.]]
        >>> print(nd_dirac_delta(11, 0.5, np.inf))
        [ 0.  0.  0.  0.  0. inf  0.  0.  0.  0.  0.]
    """
    if not n_dim:
        n_dim = fc.util.combine_iter_len((shape, position))

    # check compatibility of given parameters
    shape = fc.util.auto_repeat(shape, n_dim, check=True)
    position = fc.util.auto_repeat(position, n_dim, check=True)

    origin = fc.num.coord(
        shape, position, is_relative=rel_position, use_int=True)

    rendered = np.zeros(shape, dtype=type(value))
    rendered[tuple(slice(i, i + 1) for i in origin)] = value
    return rendered


# ======================================================================
def nd_polytope(
        shape,
        vertices,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Render a simple polytope.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        vertices (Sequence[Sequence[float]): Coordinates of the vertices.
            The values interpretation depend on `rel_sizes`.
        position (float|Sequence[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters,
            but one of `shape`, `position` must be a sequence.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `position` using `shape`.
            Uses `fc.num.grid_coord()` internally, see its `is_relative`
            parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `vertices` using `shape`.
            Uses `fc.num.coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object.
    """
    raise NotImplementedError


# ======================================================================
def ellipsoid_specs(
        n_dim,
        intensity=1.0,
        semisizes=0.25,
        shift=0.0,
        angles=0.0,
        position=0.5):
    """
    Generate the specifications for an ellipsoid.

    Args:
        n_dim (int): The number of dimensions.
        intensity (Number): The intensity of the object.
        semisizes (float|Sequence[float]): The semiaxes sizes in rel. units.
        shift (float|Sequence[float]): The shift in rel. units.
            This is relative to the smallest cuboid inscribed in `shape`.
        angles (int|float|Sequence[int|float]): The rotation angles in deg.
            These describe the rotations as generated by
            `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
        position (float|Sequence[float]|None): The position in rel. units.
            This is relative to `shape`.

    Returns:
        geom_shape (tuple): The geometric specification.
            These is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Examples:
        >>> ellipsoid_specs(2)
        (1.0, ('ellipsoid', (0.25, 0.25)), (0.0, 0.0), (0.0,), (0.5, 0.5))
    """
    n_angles = fc.num.square_size_to_num_tria(n_dim)
    geom_shape = (
        intensity,
        ('ellipsoid', fc.util.auto_repeat(semisizes, n_dim, False, True)),
        fc.util.auto_repeat(shift, n_dim, False, True),
        fc.util.auto_repeat(angles, n_angles, False, True),
        fc.util.auto_repeat(position, n_dim, False, True))
    return geom_shape


# ======================================================================
def superellipsoid_specs(
        n_dim,
        intensity=1.0,
        semisizes=0.25,
        indexes=2,
        shift=0.0,
        angles=0.0,
        position=0.5):
    """
    Generate the specifications for a superellipsoid.

    Args:
        n_dim (int): The number of dimensions.
        intensity (Number): The intensity of the object.
        semisizes (float|Sequence[float]): The semiaxes sizes in rel. units.
        indexes (float|Sequence[float]): The exponent of the summed terms.
            If 2, generates ellipsoids.
        shift (float|Sequence[float]): The shift in rel. units.
            This is relative to the smallest cuboid inscribed in `shape`.
        angles (int|float|Sequence[int|float]): The rotation angles in deg.
            These describe the rotations as generated by
            `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
        position (float|Sequence[float]|None): The position in rel. units.
            This is relative to `shape`.

    Returns:
        geom_shape (tuple): The geometric specification.
            These is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Examples:
        >>> superellipsoid_specs(2)
        (1.0, ('superellipsoid', (0.25, 0.25), (2, 2)), (0.0, 0.0), (0.0,),\
 (0.5, 0.5))
    """
    n_angles = fc.num.square_size_to_num_tria(n_dim)
    geom_shape = (
        intensity,
        ('superellipsoid',
         fc.util.auto_repeat(semisizes, n_dim, False, True),
         fc.util.auto_repeat(indexes, n_dim, False, True)),
        fc.util.auto_repeat(shift, n_dim, False, True),
        fc.util.auto_repeat(angles, n_angles, False, True),
        fc.util.auto_repeat(position, n_dim, False, True))
    return geom_shape


# ======================================================================
def cuboid_specs(
        n_dim,
        intensity=1.0,
        semisizes=0.25,
        shift=0.0,
        angles=0.0,
        position=0.5):
    """
    Generate the specifications for a cuboid.

    Args:
        n_dim (int): The number of dimensions.
        intensity (Number): The intensity of the object.
        semisizes (float|Sequence[float]): The cuboid semisides in rel. units.
        shift (float|Sequence[float]): The shift in rel. units.
            This is relative to the smallest cuboid inscribed in `shape`.
        angles (int|float|Sequence[int|float]): The rotation angles in deg.
            These describe the rotations as generated by
            `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
        position (float|Sequence[float]|None): The position in rel. units.
            This is relative to `shape`.

    Returns:
        geom_shape (tuple): The geometric specification.
            These is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Examples:
        >>> cuboid_specs(2)
        (1.0, ('cuboid', (0.25, 0.25)), (0.0, 0.0), (0.0,), (0.5, 0.5))
    """
    n_angles = fc.num.square_size_to_num_tria(n_dim)
    geom_shape = (
        intensity,
        ('cuboid', fc.util.auto_repeat(semisizes, n_dim, False, True)),
        fc.util.auto_repeat(shift, n_dim, False, True),
        fc.util.auto_repeat(angles, n_angles, False, True),
        fc.util.auto_repeat(position, n_dim, False, True))
    return geom_shape


# ======================================================================
def prism_specs(
        n_dim,
        intensity=1.0,
        semisizes=0.25,
        indexes=2,
        axis=-1,
        shift=0.0,
        angles=0.0,
        position=0.5):
    """
    Generate the specifications for an superellipsoidal prism.

    Args:
        n_dim (int): The number of dimensions.
        intensity (Number): The intensity of the object.
        semisizes (float|Sequence[float]): The semiaxes sizes in rel. units.
        indexes (float|Sequence[float]): The exponent of the summed terms.
            If 2, generates ellipsoids.
        axis (int): The axis along which to orient the prism.
        shift (float|Sequence[float]): The shift in rel. units.
            This is relative to the smallest cuboid inscribed in `shape`.
        angles (int|float|Sequence[int|float]): The rotation angles in deg.
            These describe the rotations as generated by
            `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
        position (float|Sequence[float]|None): The position in rel. units.
            This is relative to `shape`.

    Returns:
        geom_shape (tuple): The geometric specification.
            These is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Examples:
        >>> prism_specs(2)
        (1.0, ('prism', -1, (0.25,), (2,)), (0.0, 0.0), (0.0,), (0.5, 0.5))
    """
    n_angles = fc.num.square_size_to_num_tria(n_dim)
    geom_shape = (
        intensity,
        ('prism',
         axis,
         fc.util.auto_repeat(semisizes, n_dim - 1, False, True),
         fc.util.auto_repeat(indexes, n_dim - 1, False, True)),
        fc.util.auto_repeat(shift, n_dim, False, True),
        fc.util.auto_repeat(angles, n_angles, False, True),
        fc.util.auto_repeat(position, n_dim, False, True))
    return geom_shape


# ======================================================================
def gradient_specs(
        n_dim,
        intensity=1.0,
        gen_ranges=(0, 1),
        shift=0.0,
        angles=0.0,
        position=0.5):
    """
    Generate the specifications for a gradient.

    Args:
        n_dim (int): The number of dimensions.
        intensity (Number): The intensity of the object.
        gen_ranges:
        shift (float|Sequence[float]): The shift in rel. units.
            This is relative to the smallest cuboid inscribed in `shape`.
        angles (int|float|Sequence[int|float]): The rotation angles in deg.
            These describe the rotations as generated by
            `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
        position (float|Sequence[float]|None): The position in rel. units.
            This is relative to `shape`.

    Returns:
        geom_shape (tuple): The geometric specification.
            These is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Examples:
        >>> gradient_specs(2)
        (1.0, ('gradient', ((0, 1), (0, 1))), (0.0, 0.0), (0.0,), (0.5, 0.5))
    """
    n_angles = fc.num.square_size_to_num_tria(n_dim)
    geom_shape = (
        intensity,
        ('gradient', fc.util.auto_repeat(gen_ranges, n_dim, True, True)),
        fc.util.auto_repeat(shift, n_dim, False, True),
        fc.util.auto_repeat(angles, n_angles, False, True),
        fc.util.auto_repeat(position, n_dim, False, True))
    return geom_shape


# ======================================================================
def multi_render(
        shape,
        geom_shapes,
        n_dim=None,
        affine_kws=(('order', 0),),
        dtype=np.float):
    """
    Render multiple geometrical objects into a single array.

    Args:
        shape (int|Sequence[int]): The shape of the container in px.
        geom_shapes (Sequence): The geometrical shapes to render.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position)
            where:
             - intensity (Number): The intensity of the object.
             - geom_spec (tuple): The tuple describing the object.
                The first element is the name as str, all the remaining
                elements are passed to the generating function.
                Accepted values are:
                 - 'e', 'ellipsoid': uses `nd_superellipsoid()` (indexes=2)
                 - 's', 'superellipsoid': uses `nd_superellipsoid()`
                 - 'c', 'cuboid': uses `nd_cuboid()`
                 - 'p', 'prism': uses `nd_superellipsoidal_prism()`
                 - 'g', 'gradient': uses `nd_gradient()`
             - shift (float|Sequence[float]): The shift in rel. units.
                This is relative to the smallest cuboid inscribed in `shape`.
             - angles (int|float|Sequence[int|float]): The rotation angles
                in deg.
                These describe the rotations as generated by
                `flyingcircus.extra.angles2linear()` for the specified `n_dim`.
             - position (float|Sequence[float]|None): The position
                in rel. units.
                This is relative to `shape`.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the `shape` parameter,
            which must be a sequence.
        affine_kws (dict|Sequence|None): Keyword parameters for the affine.
            These parameters are passed to `scipy.ndimage.affine_transform()`
            upon application of the (eventual) rotation.
            If Sequence, must be possible to cast to dict.
            If None, no parameter is set.
        dtype (np.dtype): The Numpy data type of the rendered array.
            Note that this will be used also for the internal interpolations!

    Returns:
        rendered (np.ndarray[bool]): The rendered geometrical object(s).
    """
    # check that ellipsoids parameters have the right size
    if n_dim is None:
        n_dim = len(shape)
    else:
        shape = fc.util.auto_repeat(shape, n_dim, False, True)
    inner_shape = [min(shape)] * n_dim
    affine_kws = \
        {} if affine_kws is None else dict(affine_kws)
    rendered = np.zeros(shape, dtype=dtype)
    for intensity, geom_spec, shift, angles, position in geom_shapes:
        # generate the base geometric shape
        geom_name = geom_spec[0]
        geom_arr = None
        if geom_name in ('e', 'ellipsoid'):
            semisizes = geom_spec[1]
            geom_arr = nd_superellipsoid(inner_shape, semisizes)
        elif geom_name in ('s', 'superellipsoid'):
            semisizes, indexes = geom_spec[1:]
            geom_arr = nd_superellipsoid(inner_shape, semisizes, indexes)
        elif geom_name in ('c', 'cuboid'):
            semisizes = geom_spec[1]
            geom_arr = nd_cuboid(inner_shape, semisizes)
        elif geom_name in ('p', 'prism'):
            axis, semisizes, indexes = geom_spec[1:]
            geom_arr = nd_superellipsoidal_prism(
                inner_shape, axis % n_dim, semisizes, indexes)
        elif geom_name in ('g', 'gradient'):
            gen_ranges = geom_spec[1]
            geom_arr = np.sum(nd_gradient(inner_shape, gen_ranges), 0)
        else:
            text = ('unknown name `{geom_name}` while rendering with '
                    '`raster_geometry.multi_render`'.format(**locals()))
            warnings.warn(text)
        if position is None:
            position = 0.5
        if geom_arr is not None:
            # compute position and linear transformation matrix
            if angles is None:
                lin_mat = np.eye(n_dim)
            else:
                lin_mat = fc.num.angles2rotation(angles, n_dim)
            shift = fc.num.coord(inner_shape, shift, True, False)
            lin_mat, offset = fc.num.prepare_affine(
                inner_shape, lin_mat, shift)
            rendered += fc.num.reframe(
                intensity * sp.ndimage.affine_transform(
                    geom_arr.astype(dtype), lin_mat, offset, **affine_kws),
                shape, position)
    return rendered


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
