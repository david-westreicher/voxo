import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

import moderngl
from moderngl import Context, Texture3D
from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812
from pyglm.glm import vec3

from voxo.constants import USE_VOXEL_OBJECT_INSTANCING
from voxo.model.vox_parser import VoxModel

from .model import Model, SimplifiedModel
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

    def write(self, f: BinaryIO) -> None:
        name_bytes = self.name.encode()
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(self.rotation.to_bytes())
        f.write(self.translation.to_bytes())
        f.write(self.scale.to_bytes())

    @staticmethod
    def from_file(f: BinaryIO) -> "Object":
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


class TextureInformation:
    def __init__(self) -> None:
        self.palette_texture_data = b""
        self.material_texture_data = b""
        self.palette_row_sizes: list[int] = []
        self.material_row_sizes: list[int] = []

    def __hash__(self) -> int:
        return hash(
            (self.palette_texture_data, self.material_texture_data, self.palette_row_sizes, self.material_row_sizes)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(self, TextureInformation):
            return NotImplemented
        assert type(other) is TextureInformation
        return (
            self.palette_texture_data == other.palette_texture_data
            and self.material_texture_data == other.material_texture_data
            and self.palette_row_sizes == other.palette_row_sizes
            and self.material_row_sizes == other.material_row_sizes
        )

    def upload_to_gpu(self, ctx: Context) -> None:
        assert len(self.palette_texture_data) % (256 * 3) == 0
        rows = len(self.palette_texture_data) // (256 * 3)
        self.palette_texture = ctx.texture((256, rows), data=self.palette_texture_data, components=3, dtype="f1")
        self.palette_texture.label = "tex_world_palette_texture"
        self.palette_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.palette_texture.repeat_x = False
        self.palette_texture.repeat_y = False

        assert len(self.material_texture_data) % (256 * 4 * 2) == 0
        rows = len(self.material_texture_data) // (256 * 4 * 2)
        self.material_texture = ctx.texture((256, rows), data=self.material_texture_data, components=4, dtype="f2")
        self.material_texture.label = "tex_world_material_texture"
        self.material_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.material_texture.repeat_x = False
        self.material_texture.repeat_y = False

    def write(self, f: BinaryIO) -> None:
        # palette data
        f.write(struct.pack("<I", len(self.palette_texture_data)))
        f.write(self.palette_texture_data)
        f.write(struct.pack("<I", len(self.palette_row_sizes)))
        f.write(struct.pack(f"<{len(self.palette_row_sizes)}I", *self.palette_row_sizes))

        # material data
        f.write(struct.pack("<I", len(self.material_texture_data)))
        f.write(self.material_texture_data)
        f.write(struct.pack("<I", len(self.material_row_sizes)))
        f.write(struct.pack(f"<{len(self.material_row_sizes)}I", *self.material_row_sizes))

    @staticmethod
    def from_file(f: BinaryIO) -> "TextureInformation":
        texture_information = TextureInformation()
        # palette data
        palette_data_len, *_ = struct.unpack("<I", f.read(4))
        texture_information.palette_texture_data = f.read(palette_data_len)
        palette_rows, *_ = struct.unpack("<I", f.read(4))
        texture_information.palette_row_sizes = list(struct.unpack(f"<{palette_rows}I", f.read(4 * palette_rows)))

        # material data
        material_data_len, *_ = struct.unpack("<I", f.read(4))
        texture_information.material_texture_data = f.read(material_data_len)
        material_rows, *_ = struct.unpack("<I", f.read(4))
        texture_information.material_row_sizes = list(struct.unpack(f"<{material_rows}I", f.read(4 * material_rows)))
        return texture_information

    @staticmethod
    def from_models(models: list[Model]) -> tuple["TextureInformation", dict[bytes, int], dict[bytes, int]]:
        texture_information = TextureInformation()
        palettes = [model.palette_data for model in models]
        materials = [model.material_data for model in models]
        texture_information.palette_texture_data, palette_row_mapping, texture_information.palette_row_sizes = (
            generate_palette_texture(palettes)
        )
        texture_information.material_texture_data, material_row_mapping, texture_information.material_row_sizes = (
            generate_material_texture(materials)
        )
        return texture_information, palette_row_mapping, material_row_mapping


@dataclass(kw_only=True)
class VoxelObject(Object):
    model: SimplifiedModel
    texture_information: TextureInformation
    last_frame_transform: glm.mat4x4 = field(default_factory=lambda: glm.identity(glm.mat4x4))
    is_dirty = True
    _voxel_texture: Texture3D | None = None

    def __post_init__(self) -> None:
        if not self.name:
            global OBJECT_ID_COUNTER  # noqa: PLW0603
            self.name = f"{self.model.name}_{OBJECT_ID_COUNTER}"
            OBJECT_ID_COUNTER += 1
        super().__post_init__()
        self._center_translation = -cast("glm.vec3", glm.ceil(glm.vec3(self.model.opengl_dimensions) * 0.5))
        self._center_translation.y = 0

    def upload_to_gpu(self, ctx: Context) -> None:
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

        if USE_VOXEL_OBJECT_INSTANCING:
            self.voxel_texture_handle = self._voxel_texture.get_handle(resident=True)

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
    def transform(self) -> Mat4:
        return cast(
            "Mat4",
            glm.translate(self.translation)
            @ glm.mat4_cast(self.rotation)
            @ glm.scale(self.scale)
            @ glm.translate(self._center_translation),
        )

    def write(self, f: BinaryIO) -> None:
        super().write(f)
        self.model.write(f)

    @staticmethod
    def from_file(f: BinaryIO, texture_information: TextureInformation) -> "VoxelObject":  # type:ignore[override]
        obj = Object.from_file(f)
        model = SimplifiedModel.from_file(f, obj.name)
        return VoxelObject(
            name=obj.name,
            texture_information=texture_information,
            rotation=obj.rotation,
            translation=obj.translation,
            scale=obj.scale,
            model=model,
        )


def generate_palette_texture(palettes: list[bytes]) -> tuple[bytes, dict[bytes, int], list[int]]:
    unique_palette_data = sorted(set(palettes))
    palette_row_mapping = {e: i for i, e in enumerate(unique_palette_data)}
    full_palette_data: list[bytes] = []
    row_sizes = []
    for palette in unique_palette_data:
        full_palette_data.append(palette)
        remaining_length_for_row = 256 * 3 - len(palette)
        row_sizes.append(len(palette) // 3 - 1)
        assert remaining_length_for_row >= 0
        full_palette_data.append(b"\x00" * remaining_length_for_row)
    return b"".join(full_palette_data), palette_row_mapping, row_sizes


def generate_material_texture(materials: list[bytes]) -> tuple[bytes, dict[bytes, int], list[int]]:
    unique_material_data = sorted(set(materials))
    material_row_mapping = {e: i for i, e in enumerate(unique_material_data)}
    full_material_data: list[bytes] = []
    row_sizes = []
    for material in unique_material_data:
        full_material_data.append(material)
        remaining_length_for_row = 256 * 4 * 2 - len(material)
        row_sizes.append(len(material) // (4 * 2) - 1)
        assert remaining_length_for_row >= 0
        full_material_data.append(b"\x00" * remaining_length_for_row)
    return b"".join(full_material_data), material_row_mapping, row_sizes


class World:
    def __init__(self) -> None:
        self.voxel_objects: list[VoxelObject] = []
        self.texture_information = TextureInformation()

    @staticmethod
    def from_vox_models(vox_models: list[VoxModel]) -> "World":
        world = World()
        models = [vox_model.to_model() for vox_model in vox_models]
        world.texture_information, palette_row_mapping, material_row_mapping = TextureInformation.from_models(models)
        for vox_model, model in zip(vox_models, models, strict=True):
            palette_row = palette_row_mapping[model.palette_data]
            material_row = material_row_mapping[model.material_data]
            world.voxel_objects.append(
                VoxelObject(
                    name=vox_model.shape_name,
                    texture_information=world.texture_information,
                    model=model.simplify(palette_row, material_row),
                    rotation=vox_model.rotation,
                    translation=vox_model.translation,
                )
            )
        return world

    @staticmethod
    def from_file(world_file: Path) -> "World":
        world = World()
        with world_file.open("rb", buffering=1024 * 1024 * 10) as f:
            world.texture_information = TextureInformation.from_file(f)
            obj_num, *_ = struct.unpack("<I", f.read(4))
            for _ in range(obj_num):
                world.voxel_objects.append(VoxelObject.from_file(f, world.texture_information))
            assert f.read(1) == b""  # EOF
        return world

    def write(self, world_file: Path) -> None:
        with world_file.open("wb", buffering=1024 * 1024 * 10) as f:
            self.texture_information.write(f)

            f.write(struct.pack("<I", len(self.voxel_objects)))
            for vox_obj in self.voxel_objects:
                vox_obj.write(f)
