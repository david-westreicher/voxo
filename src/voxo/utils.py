import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import TypeVar, cast

import moderngl as mlg
import numpy as np
from moderngl_window.geometry import AttributeNames
from moderngl_window.opengl.vao import VAO
from pyglm import glm

T = TypeVar("T")


@dataclass
class Plane:
    normal: glm.vec3
    d: float


@dataclass
class Sphere:
    radius: float
    center: glm.vec3


@dataclass
class Ray:
    origin: glm.vec3
    direction: glm.vec3


def sphere_in_frustum(planes: list[Plane], sphere: Sphere) -> bool:
    for p in planes:
        distance = glm.dot(p.normal, sphere.center) + p.d
        if distance < -sphere.radius:
            return False
    return True


def compute_frustum_planes(view: glm.mat4x4, projection: glm.mat4x4) -> list[Plane]:
    vp = projection @ view
    return [
        Plane(glm.vec3(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0]), vp[3][3] + vp[3][0]),  # Left
        Plane(glm.vec3(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0]), vp[3][3] - vp[3][0]),  # Right
        Plane(glm.vec3(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1]), vp[3][3] + vp[3][1]),  # Bottom
        Plane(glm.vec3(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1]), vp[3][3] - vp[3][1]),  # Top
        Plane(glm.vec3(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2]), vp[3][3] + vp[3][2]),  # Near
        Plane(glm.vec3(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2]), vp[3][3] - vp[3][2]),  # Far
    ]


def frustum_cull_spheres(view: glm.mat4x4, projection: glm.mat4x4, spheres: Sequence[Sphere]) -> Iterable[bool]:
    planes = compute_frustum_planes(view, projection)
    for sphere in spheres:
        yield sphere_in_frustum(planes, sphere)


def ray_sphere_intersection(ray: Ray, sphere: Sphere) -> tuple[bool, float]:
    oc = sphere.center - ray.origin
    t = glm.dot(oc, ray.direction)
    if t < 0:
        return False, 0
    closest = ray.origin + t * ray.direction
    dist2 = glm.length2(closest - sphere.center)  # type:ignore[call-overload]
    if dist2 > sphere.radius * sphere.radius:
        return False, 0

    # Calculate exact intersection distance
    thc = glm.sqrt(sphere.radius * sphere.radius - dist2)
    hit_distance = t - thc
    return True, hit_distance


def compute_camera_ray(ndc: glm.vec2, proj: glm.mat4x4, view: glm.mat4x4) -> Ray:
    clip = glm.vec4(ndc.x, ndc.y, -1.0, 1.0)
    eye = cast("glm.vec4", glm.inverse(proj) * clip)
    eye = glm.vec4(eye.x, eye.y, -1.0, 0.0)
    camera_pos = glm.vec3(glm.inverse(view)[3])
    direction = glm.vec3(glm.inverse(view) * eye)
    return Ray(camera_pos, glm.normalize(direction))


def chunk_iters[T](iterator: Iterable[T], size: int) -> Iterator[list[T]]:
    it = iter(iterator)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def cone(  # noqa: PLR0913
    angle: float = 30.0,
    penumbra: float = 5.0,
    max_distance: float = 10.0,
    sectors: int = 32,
    rings: int = 16,
    name: str | None = None,
    attr_names: type[AttributeNames] = AttributeNames,
) -> VAO:
    """
    Creates a spotlight volume:
        - cone along +Y
        - rounded hemispherical end cap

    The mesh represents the outer cone boundary.
    Inner cone / penumbra interpolation belongs in the shader.

    Args:
        angle:
            Inner cone full angle in degrees.

        penumbra:
            Additional outer cone angle in degrees.

        max_distance:
            Light reach.
    """

    vertices_l = []

    # Outer cone defines geometry coverage
    outer_angle = math.radians((angle + penumbra) * 0.5)

    # Radius where cone meets hemisphere
    cone_radius = math.tan(outer_angle) * max_distance

    cone_rings = rings
    cap_rings = max(4, rings // 2)

    total_rings = cone_rings + cap_rings

    for r in range(total_rings):
        if r < cone_rings:
            # -------------------------
            # Cone section
            # -------------------------
            t = r / (cone_rings - 1)
            y = max_distance * t
            radius = cone_radius * t

        else:
            # -------------------------
            # Hemisphere cap
            # -------------------------
            t = (r - cone_rings + 1) / cap_rings

            theta = t * (math.pi * 0.5)

            radius = cone_radius * math.cos(theta)
            y = max_distance + cone_radius * math.sin(theta)

        for s in range(sectors):
            a = 2.0 * math.pi * s / (sectors - 1)

            x = math.cos(a) * radius
            z = math.sin(a) * radius

            vertices_l.extend([x, y, z])

    indices = []

    for r in range(total_rings - 1):
        for s in range(sectors - 1):
            a = r * sectors + s
            b = (r + 1) * sectors + s
            c = (r + 1) * sectors + s + 1
            d = r * sectors + s + 1

            indices.extend([a, b, d, d, b, c])

    vao = VAO(name or "cone", mode=mlg.TRIANGLES)
    vao.buffer(np.array(vertices_l, dtype=np.float32), "3f", [attr_names.POSITION])
    vao.index_buffer(np.array(indices, dtype=np.uint32), index_element_size=4)
    return vao


def hemisphere(
    radius: float = 0.5,
    sectors: int = 32,
    rings: int = 16,
    name: str | None = None,
    attr_names: type[AttributeNames] = AttributeNames,
) -> VAO:
    """Creates an upper hemisphere with a planar base."""

    inverse_radius = 1.0 / (rings - 1)
    inverse_sectors = 1.0 / (sectors - 1)

    vertices_l = []
    indices = []

    # -------------------------
    # Curved hemisphere surface
    # -------------------------

    for r in range(rings):
        phi = (math.pi / 2) * r * inverse_radius

        y = math.sin(phi)
        ring_radius = math.cos(phi)

        for s in range(sectors):
            a = 2.0 * math.pi * s * inverse_sectors
            x = math.cos(a) * ring_radius
            z = math.sin(a) * ring_radius
            vertices_l.extend([x * radius, y * radius, z * radius])

    # Curved surface indices
    for r in range(rings - 1):
        for s in range(sectors - 1):
            a = r * sectors + s
            b = (r + 1) * sectors + s
            c = (r + 1) * sectors + s + 1
            d = r * sectors + s + 1

            indices.extend([a, b, d, d, b, c])

    # -------------------------
    # Planar base
    # -------------------------
    #
    # Duplicate equator ring because normals differ:
    # hemisphere normal = outward
    # base normal      = downward

    base_start = len(vertices_l) // 3

    for s in range(sectors):
        a = 2.0 * math.pi * s * inverse_sectors

        x = math.cos(a)
        z = math.sin(a)

        vertices_l.extend([x * radius, 0.0, z * radius])

    # Base center vertex
    center_index = len(vertices_l) // 3

    vertices_l.extend([0.0, 0.0, 0.0])

    # Triangle fan
    for s in range(sectors - 1):
        indices.extend([center_index, base_start + s, base_start + s + 1])

    vao = VAO(name or "hemisphere", mode=mlg.TRIANGLES)
    vao.buffer(np.array(vertices_l, dtype=np.float32), "3f", [attr_names.POSITION])
    vao.index_buffer(np.array(indices, dtype=np.uint32), index_element_size=4)
    return vao
