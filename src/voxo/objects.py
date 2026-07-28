import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

import moderngl
from moderngl import Context, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812
from pyglm.glm import vec3

from .model import Model
from .utils import Sphere

OBJECT_ID_COUNTER = 0


@dataclass
class Object:
    geometry: VAO | None = None
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

    def serialize(self, f: BinaryIO) -> None:
        name_bytes = self.name.encode()
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(self.rotation.to_bytes())
        f.write(self.translation.to_bytes())
        f.write(self.scale.to_bytes())

    @staticmethod
    def deserialize(f: BinaryIO) -> "Object":
        name_len, *_ = struct.unpack("<I", f.read(4))
        name = f.read(name_len).decode()
        rotation = Quat.from_bytes(f.read(16))
        translation = vec3.from_bytes(f.read(12))
        scale = vec3.from_bytes(f.read(12))
        return Object(name=name, rotation=rotation, translation=translation, scale=scale)


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
    geometry: VAO | None = None
    last_frame_transform: glm.mat4x4 = field(default_factory=lambda: glm.identity(glm.mat4x4))
    is_dirty = True
    _voxel_texture: Texture3D | None = None
    _palette_texture: Texture | None = None

    def __post_init__(self) -> None:
        if not self.name:
            global OBJECT_ID_COUNTER  # noqa: PLW0603
            self.name = f"{self.model.name}_{OBJECT_ID_COUNTER}"
            OBJECT_ID_COUNTER += 1
        super().__post_init__()
        self._center_translation: glm.vec3 = -cast("glm.vec3", glm.ceil(glm.vec3(self.model.opengl_dimensions) * 0.5))
        self._center_translation.y = 0

    def upload_to_gpu(self, ctx: Context) -> None:
        self.geometry = geometry.cube(
            size=self.model.opengl_dimensions,
            center=(glm.vec3(self.model.opengl_dimensions) * 0.5).to_tuple(),
        )
        self._voxel_texture = ctx.texture3d(
            self.model.opengl_dimensions,
            data=self.model.voxel_data,
            components=1,
            alignment=1,
            dtype="u1",
            create_mip_maps=True,
        )
        self._voxel_texture.label = f"tex3d_model_{self.name}"
        self._voxel_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._voxel_texture.repeat_x = False
        self._voxel_texture.repeat_y = False
        self._voxel_texture.repeat_z = False

        palette = self.model.palette_data
        self._palette_texture = ctx.texture((len(palette) // 3, 1), data=palette, components=3, dtype="f1")
        self._palette_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._palette_texture.repeat_x = False
        self._palette_texture.repeat_y = False

        material = self.model.material_data
        self._material_texture = ctx.texture((len(material) // 8, 1), data=material, components=4, dtype="f2")
        self._material_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._material_texture.repeat_x = False
        self._material_texture.repeat_y = False

    @property
    def center(self) -> glm.vec3:
        dim = glm.vec4(glm.vec3(self.model.opengl_dimensions) * 0.5, 1.0)  # type:ignore[call-overload]
        pos = cast("glm.vec4", self.transform * dim)
        pos = pos / pos.w
        return glm.vec3(pos)

    @property
    def aabb(self) -> tuple[glm.vec3, glm.vec3]:
        w, h, d = self.model.opengl_dimensions
        corners = [
            glm.vec4(0, 0, 0, 1),
            glm.vec4(w, 0, 0, 1),
            glm.vec4(0, h, 0, 1),
            glm.vec4(w, h, 0, 1),
            glm.vec4(0, 0, d, 1),
            glm.vec4(w, 0, d, 1),
            glm.vec4(0, h, d, 1),
            glm.vec4(w, h, d, 1),
        ]

        aabb_min = glm.vec3(float("inf"))
        aabb_max = glm.vec3(float("-inf"))

        for corner in corners:
            world = glm.vec3(self.transform * corner)

            aabb_min = cast("glm.vec3", glm.min(aabb_min, world))
            aabb_max = cast("glm.vec3", glm.max(aabb_max, world))

        return aabb_min, aabb_max

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
    def material_texture(self) -> Texture:
        assert self._material_texture
        return self._material_texture

    @property
    def transform(self) -> Mat4:
        return cast(
            "Mat4",
            glm.translate(self.translation)
            @ glm.mat4_cast(self.rotation)
            @ glm.scale(self.scale)
            @ glm.translate(self._center_translation),
        )

    def serialize(self, f: BinaryIO) -> None:
        super().serialize(f)
        self.model.serialize(f)

    @staticmethod
    def deserialize(f: BinaryIO) -> "VoxelObject":
        obj = Object.deserialize(f)
        model = Model.deserialize(f, obj.name)
        return VoxelObject(
            name=obj.name,
            rotation=obj.rotation,
            translation=obj.translation,
            scale=obj.scale,
            model=model,
        )


class World:
    def write(self, world_file: Path, vox_objects: Sequence[VoxelObject]) -> None:
        with world_file.open("wb", buffering=1024 * 1024 * 10) as f:
            f.write(struct.pack("<I", len(vox_objects)))
            for vox_obj in vox_objects:
                vox_obj.serialize(f)

    def read(self, world_file: Path) -> Iterable[VoxelObject]:
        with world_file.open("rb", buffering=1024 * 1024 * 10) as f:
            obj_num, *_ = struct.unpack("<I", f.read(4))
            for _ in range(obj_num):
                yield VoxelObject.deserialize(f)
            assert f.read(1) == b""  # EOF
