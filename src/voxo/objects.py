from dataclasses import dataclass, field
from typing import cast

import moderngl
from moderngl import Context, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812

from .model import Model
from .utils import Sphere

OBJECT_ID_COUNTER = 0


@dataclass
class Object:
    geometry: VAO
    name: str = ""
    visible: bool = True
    rotation: Quat = field(default=glm.quat())
    translation: glm.vec3 = field(default=glm.vec3(0.0))
    scale: glm.vec3 = field(default=glm.vec3(1.0))

    def __post_init__(self) -> None:
        if not self.name:
            global OBJECT_ID_COUNTER  # noqa: PLW0603
            self.name = f"obj-{OBJECT_ID_COUNTER:04d}"
            OBJECT_ID_COUNTER += 1

    @property
    def transform(self) -> Mat4:
        return cast("Mat4", glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(self.scale))

    def rotate(self, angle: float, axis: glm.vec3) -> None:
        self.rotation = cast("Quat", glm.rotate(self.rotation, angle, axis))


@dataclass(init=False, kw_only=True)
class Light(Object):
    color: glm.vec3
    intensity: float = 1.0
    radius: float = 1.0

    def __init__(self, radius: float = 1.0, light_color: glm.vec3 | None = None, intensity: float = 1.0) -> None:
        global OBJECT_ID_COUNTER  # noqa: PLW0603
        super().__init__(geometry.sphere(1.0), name=f"light_{OBJECT_ID_COUNTER}")
        OBJECT_ID_COUNTER += 1
        self.color = light_color or glm.vec3(1.0)
        self.radius = radius
        self.intensity = intensity

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
        rot = glm.inverse(glm.quatLookAt(self.direction, glm.vec3(0, -1, 0)))
        return cast("Mat4", glm.translate(self.translation) @ rot @ glm.scale(glm.vec3(self.radius)))


@dataclass(kw_only=True)
class VoxelObject(Object):
    model: Model
    geometry: VAO = field(default_factory=lambda: geometry.cube(size=(1, 1, 1)))
    last_frame_transform: glm.mat4x4 = field(default_factory=lambda: glm.identity(glm.mat4x4))
    _voxel_texture: Texture3D | None = None
    _palette_texture: Texture | None = None

    def __post_init__(self) -> None:
        if not self.name:
            global OBJECT_ID_COUNTER  # noqa: PLW0603
            self.name = f"{self.model.path.with_suffix('').name}_{OBJECT_ID_COUNTER}"
            OBJECT_ID_COUNTER += 1
        super().__post_init__()
        self.geometry = geometry.cube(
            size=self.model.opengl_dimensions,
            center=(glm.vec3(self.model.opengl_dimensions) * 0.5).to_tuple(),
        )
        self._center_translation: glm.vec3 = cast("glm.vec3", glm.floor(glm.vec3(self.model.opengl_dimensions) * 0.5))
        self._center_translation.y = 0

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
    def center(self) -> glm.vec3:
        dim = glm.vec4(glm.vec3(self.model.opengl_dimensions) * 0.5, 1.0)  # type:ignore[call-overload]
        pos = cast("glm.vec4", self.transform * dim)
        pos = pos / pos.w
        return glm.vec3(pos)

    @property
    def bounding_sphere(self) -> Sphere:
        radius = glm.length(glm.vec3(self.model.opengl_dimensions) * 0.8)
        return Sphere(radius=radius, center=self.center)

    @property
    def voxel_texture(self) -> Texture3D:
        assert self._voxel_texture
        return self._voxel_texture

    @property
    def palette_texture(self) -> Texture:
        assert self._palette_texture
        return self._palette_texture

    @property
    def transform(self) -> Mat4:
        return cast(
            "Mat4",
            glm.translate(self.translation - self._center_translation)
            @ glm.mat4_cast(self.rotation)
            @ glm.scale(self.scale),
        )
