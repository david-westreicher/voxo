from dataclasses import dataclass, field
from typing import cast

import moderngl
from moderngl import Context, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812
from pyglm.glm import vec3 as Vec3  # noqa: N812

from .model import Model


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


@dataclass(kw_only=True)
class VoxelObject(Object):
    model: Model
    geometry: VAO = field(default_factory=lambda: geometry.cube(size=(1, 1, 1)))
    _voxel_texture: Texture3D | None = None
    _palette_texture: Texture | None = None

    def __post_init__(self) -> None:
        self.geometry = geometry.cube(size=self.model.opengl_dimensions)

    def upload_to_gpu(self, ctx: Context) -> None:
        self._voxel_texture = ctx.texture3d(
            self.model.opengl_dimensions,
            data=self.model.generate_voxel_data(),
            components=1,
            alignment=1,
            dtype="u1",
            create_mip_maps=True,
        )
        self._voxel_texture.label = f"tex3d_model_{self.model.name}"
        self._voxel_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._voxel_texture.repeat_x = False
        self._voxel_texture.repeat_y = False
        self._voxel_texture.repeat_z = False

        palette = self.model.generate_palette_data()
        self._palette_texture = ctx.texture((len(palette) // 3, 1), data=palette, components=3, dtype="f1")
        self._palette_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._palette_texture.repeat_x = False
        self._palette_texture.repeat_y = False

    @property
    def voxel_texture(self) -> Texture3D:
        assert self._voxel_texture
        return self._voxel_texture

    @property
    def palette_texture(self) -> Texture:
        assert self._palette_texture
        return self._palette_texture
