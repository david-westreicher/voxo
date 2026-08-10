import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import islice
from pathlib import Path
from typing import TypeVar, cast

import imageio as imageio_base
import imageio.v3 as imageio
import moderngl as mlg
import numpy as np
from moderngl import Context, Texture
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
    angle: float = 5.0,
    max_distance: float = 10.0,
    sectors: int = 32,
    rings: int = 16,
    name: str | None = None,
    attr_names: type[AttributeNames] = AttributeNames,
) -> VAO:
    vertices_l = []

    outer_angle = math.radians((angle) * 0.5)
    cone_radius = math.tan(outer_angle) * max_distance

    cone_rings = rings
    cap_rings = 1

    total_rings = cone_rings + cap_rings

    for r in range(total_rings):
        if r < cone_rings:
            # cone
            t = r / (cone_rings - 1)
            y = max_distance * t
            radius = cone_radius * t

        else:
            # cap
            t = (r - cone_rings + 1) / cap_rings

            theta = t * (math.pi * 0.5)

            radius = cone_radius * math.cos(theta)
            y = max_distance

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

    for r in range(rings):
        phi = (math.pi / 2) * r * inverse_radius

        y = math.sin(phi)
        ring_radius = math.cos(phi)

        for s in range(sectors):
            a = 2.0 * math.pi * s * inverse_sectors
            x = math.cos(a) * ring_radius
            z = math.sin(a) * ring_radius
            vertices_l.extend([x * radius, y * radius, z * radius])

    for r in range(rings - 1):
        for s in range(sectors - 1):
            a = r * sectors + s
            b = (r + 1) * sectors + s
            c = (r + 1) * sectors + s + 1
            d = r * sectors + s + 1

            indices.extend([a, b, d, d, b, c])

    base_start = len(vertices_l) // 3

    for s in range(sectors):
        a = 2.0 * math.pi * s * inverse_sectors

        x = math.cos(a)
        z = math.sin(a)

        vertices_l.extend([x * radius, 0.0, z * radius])

    center_index = len(vertices_l) // 3

    vertices_l.extend([0.0, 0.0, 0.0])

    for s in range(sectors - 1):
        indices.extend([center_index, base_start + s, base_start + s + 1])

    vao = VAO(name or "hemisphere", mode=mlg.TRIANGLES)
    vao.buffer(np.array(vertices_l, dtype=np.float32), "3f", [attr_names.POSITION])
    vao.index_buffer(np.array(indices, dtype=np.uint32), index_element_size=4)
    return vao


def hdr_texture(path: Path, ctx: Context, post_processing: Callable[[np.ndarray], np.ndarray] | None = None) -> Texture:
    imageio_base.plugins.freeimage.download()
    image = imageio.imread(path).astype(np.float16)[::-1, :, :]
    if post_processing:
        image = post_processing(image)
    image = np.ascontiguousarray(image)
    size = tuple(image.shape[:-1])[::-1]
    return ctx.texture(size, components=image.shape[-1], data=image, dtype="f2")


class Timer:
    def __init__(self) -> None:
        self.frame_counter = 0

    @property
    def time(self) -> int:
        return self.frame_counter

    def tick(self) -> None:
        self.frame_counter += 1

    @cache  # noqa: B019
    def global_timer() -> "Timer":
        return Timer()
