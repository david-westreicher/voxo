from dataclasses import dataclass, field
from typing import cast

from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812
from pyglm.glm import vec3 as Vec3  # noqa: N812


@dataclass
class Object:
    geometry: VAO
    rotation: Quat = field(default=glm.quat())
    translation: Vec3 = field(default=glm.vec3(0.0))
    scale: Vec3 = field(default=glm.vec3(1.0))

    @property
    def transform(self) -> Mat4:
        return cast("Mat4", glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(self.scale))

    def rotate(self, angle: float, axis: Vec3) -> None:
        self.rotation = cast("Quat", glm.rotate(self.rotation, angle, axis))


@dataclass(init=False, kw_only=True)
class Light(Object):
    color: glm.vec3
    radius: float = 1.0

    def __init__(self, radius: float, light_color: glm.vec3) -> None:
        super().__init__(geometry.sphere(1.0))
        self.color = light_color
        self.radius = radius

    @property
    def transform(self) -> Mat4:
        return cast(
            "Mat4", glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(glm.vec3(self.radius))
        )


@dataclass(init=False, kw_only=True)
class Sun(Object):
    color: glm.vec3
    direction: glm.vec3
    radius: float = 0.1

    def __init__(self) -> None:
        super().__init__(geometry.cube(size=(1.0, 10.0, 1.0)))
        self.color = glm.vec3(1.0, 0.95, 0.85) * 2.5
        self.direction = glm.normalize(glm.vec3(1.0, 1.0, 1.0))

    @property
    def transform(self) -> Mat4:
        rot = glm.inverse(glm.quatLookAt(self.direction, glm.vec3(0, 1, 0)))
        return cast("Mat4", glm.translate(self.translation) @ rot @ glm.scale(glm.vec3(self.radius)))
