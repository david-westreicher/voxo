from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from pyglm import glm


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
    dist2 = glm.length2(closest - sphere.center)
    if dist2 > sphere.radius * sphere.radius:
        return False, None

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
