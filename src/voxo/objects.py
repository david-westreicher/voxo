import struct
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import moderngl
from moderngl import Context, Texture3D
from moderngl_window import geometry
from moderngl_window.opengl.vao import VAO
from pyglm import glm
from pyglm.glm import mat4x4, vec3
from pyglm.glm import quat as Quat  # noqa: N812

from .constants import (
    LIGHT_TYPE_NUM_AREA,
    LIGHT_TYPE_NUM_CONE,
    LIGHT_TYPE_NUM_SPHERE,
    MAX_VOXEL_OBJECTS,
    USE_VOXEL_OBJECT_INSTANCING,
)
from .model import Model, SimplifiedModel
from .model.level_parser import VoxLight, VoxWater
from .model.vox_parser import VoxModel
from .utils import Sphere, Timer, cone, hemisphere

OBJECT_ID_COUNTER = 0


@dataclass
class Object:
    geometry: VAO | None = None
    name: str = ""
    visible: bool = True
    last_frame_update: int = 0
    _rotation: Quat = field(default=glm.quat())
    _translation: glm.vec3 = field(default=glm.vec3(0.0))
    _scale: glm.vec3 = field(default=glm.vec3(1.0))

    @property
    def rotation(self) -> Quat:
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quat) -> None:
        self._rotation = value
        self.last_frame_update = self.timer.time

    @property
    def translation(self) -> glm.vec3:
        return self._translation

    @translation.setter
    def translation(self, value: glm.vec3) -> None:
        self._translation = value
        self.last_frame_update = self.timer.time

    @property
    def scale(self) -> glm.vec3:
        return self._scale

    @scale.setter
    def scale(self, value: glm.vec3) -> None:
        self._scale = value
        self.last_frame_update = self.timer.time

    def __post_init__(self) -> None:
        if not self.name:
            global OBJECT_ID_COUNTER  # noqa: PLW0603
            self.name = f"obj-{OBJECT_ID_COUNTER:04d}"
            OBJECT_ID_COUNTER += 1
        self.timer = Timer.global_timer()

    @property
    def transform(self) -> mat4x4:
        return glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(self.scale)  # type:ignore[return-value]

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
        return Object(name=name, _rotation=rotation, _translation=translation, _scale=scale)


@dataclass(kw_only=True)
class Light(Object):
    color: glm.vec3 = field(default_factory=lambda: glm.vec3(1.0))
    proxy_geometry: VAO | None = None
    intensity: float = 1.0
    reach: float = 1.0
    unshadowed: float = 0.0

    def upload_to_gpu(self) -> None: ...

    @property
    def proxy_transform(self) -> mat4x4:
        return self.transform

    @staticmethod
    def from_vox_light(vox_light: VoxLight) -> "Light":
        light: Light | None = None
        if vox_light.light_type == "sphere":
            light = SphereLight(vox_light.light_size)
        elif vox_light.light_type == "cone":
            light = ConeLight(vox_light.penumbra)
        elif vox_light.light_type == "area":
            light = AreaLight(vox_light.size * 10.0)
        else:
            raise NotImplementedError(vox_light)
        assert light
        light.name += f"_{vox_light.name}"
        light.translation = vox_light.translation
        light.rotation = vox_light.rotation
        light.color = vox_light.color
        light.intensity = 1  # vox_light.scale
        light.reach = vox_light.reach * 10.0
        light.unshadowed = vox_light.unshadowed * 20.0 or 5.0
        return light

    def write(self, f: BinaryIO) -> None:
        super().write(f)
        f.write(self.color.to_bytes())
        f.write(struct.pack("f", self.intensity))
        f.write(struct.pack("f", self.reach))
        f.write(struct.pack("f", self.unshadowed))

    @staticmethod
    def from_file(f: BinaryIO) -> "Light":
        light_type, *_ = struct.unpack("<I", f.read(4))
        obj = Object.from_file(f)
        color = glm.vec3.from_bytes(f.read(12))
        intensity, *_ = struct.unpack("f", f.read(4))
        reach, *_ = struct.unpack("f", f.read(4))
        unshadowed, *_ = struct.unpack("f", f.read(4))
        if light_type == LIGHT_TYPE_NUM_SPHERE:
            light = SphereLight.from_file(f)
        elif light_type == LIGHT_TYPE_NUM_CONE:
            light = ConeLight.from_file(f)
        elif light_type == LIGHT_TYPE_NUM_AREA:
            light = AreaLight.from_file(f)
        else:
            raise NotImplementedError
        light.name = obj.name
        light.translation = obj.translation
        light.rotation = obj.rotation
        light.color = color
        light.intensity = intensity
        light.reach = reach
        light.unshadowed = unshadowed
        return light


@dataclass(init=False, kw_only=True)
class SphereLight(Light):
    light_size: float

    def __init__(self, light_size: float = 0.1) -> None:
        global OBJECT_ID_COUNTER  # noqa: PLW0603
        super().__init__(name=f"sphere_light_{OBJECT_ID_COUNTER}")
        OBJECT_ID_COUNTER += 1
        self.light_size = light_size

    def upload_to_gpu(self) -> None:
        self.geometry = geometry.sphere(1.0)
        self.proxy_geometry = self.geometry

    @property
    def transform(self) -> mat4x4:
        return glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(glm.vec3(self.reach))  # type:ignore[return-value]

    def write(self, f: BinaryIO) -> None:
        f.write(struct.pack("<I", LIGHT_TYPE_NUM_SPHERE))
        super().write(f)
        f.write(struct.pack("<f", self.light_size))

    @staticmethod
    def from_file(f: BinaryIO) -> "Light":
        light_size, *_ = struct.unpack("<f", f.read(4))
        return SphereLight(light_size)


@dataclass(init=False, kw_only=True)
class ConeLight(Light):
    penumbra: float

    def __init__(self, penumbra: float) -> None:
        global OBJECT_ID_COUNTER  # noqa: PLW0603
        super().__init__(name=f"cone_light_{OBJECT_ID_COUNTER}")
        OBJECT_ID_COUNTER += 1
        self.penumbra = penumbra

    def upload_to_gpu(self) -> None:
        self.geometry = geometry.cube(size=(0.1, 0.1, 2.0), center=(0, 0, 1))
        self.proxy_geometry = cone(angle=self.penumbra * 2.0, max_distance=1.0, rings=2)

    @property
    def transform(self) -> mat4x4:
        return glm.translate(self.translation) @ glm.mat4_cast(self.rotation)  # type:ignore[return-value]

    @property
    def proxy_transform(self) -> mat4x4:
        return (
            glm.translate(self.translation)  # type:ignore[return-value]
            @ glm.mat4_cast(self.rotation)
            @ glm.rotate(glm.radians(90), glm.vec3(1, 0, 0))
            @ glm.scale(glm.vec3(self.reach))
        )

    @property
    def direction(self) -> glm.vec3:
        return glm.normalize(self.rotation @ glm.vec3(0, 0, -1))  # type:ignore[return-value]

    def write(self, f: BinaryIO) -> None:
        f.write(struct.pack("<I", LIGHT_TYPE_NUM_CONE))
        super().write(f)
        f.write(struct.pack("f", self.penumbra))

    @staticmethod
    def from_file(f: BinaryIO) -> "Light":
        penumbra, *_ = struct.unpack("f", f.read(4))
        return ConeLight(penumbra=penumbra)


@dataclass(init=False, kw_only=True)
class AreaLight(Light):
    size: glm.vec2

    def __init__(self, size: glm.vec2) -> None:
        global OBJECT_ID_COUNTER  # noqa: PLW0603
        super().__init__(name=f"area_light_{OBJECT_ID_COUNTER}")
        OBJECT_ID_COUNTER += 1
        self.size = size

    def upload_to_gpu(self) -> None:
        self.geometry = geometry.cube(size=(1.0, 0.001, 1.0))
        self.proxy_geometry = hemisphere(1.0)

    @property
    def transform(self) -> mat4x4:
        return (
            glm.translate(self.translation)  # type:ignore[return-value]
            @ glm.mat4_cast(self.rotation)
            @ glm.rotate(glm.radians(90), glm.vec3(1, 0, 0))
            @ glm.scale(glm.vec3(self.size.x, 1.0, self.size.y))
        )

    @property
    def proxy_transform(self) -> mat4x4:
        return (
            glm.translate(self.translation)  # type:ignore[return-value]
            @ glm.mat4_cast(self.rotation)
            @ glm.rotate(glm.radians(90), glm.vec3(1, 0, 0))
            @ glm.scale(glm.vec3(self.reach))
        )

    @property
    def area_light_matrix(self) -> glm.mat3x3:
        center = self.translation
        size = self.size
        normal = glm.mat4_cast(self.rotation) @ glm.rotate(glm.radians(90), glm.vec3(1, 0, 0)) @ glm.vec3(0, 1, 0)
        normal = glm.normalize(normal)  # type:ignore[call-overload]
        tangent = glm.vec3(0, 1, 0) if abs(normal.y) < 0.999 else glm.vec3(1, 0, 0)

        right = glm.normalize(glm.cross(tangent, normal))
        up = glm.normalize(glm.cross(normal, right))
        right *= size.x
        up *= size.y
        return glm.mat3(right, up, center)  # type:ignore[call-overload, no-any-return]

    def write(self, f: BinaryIO) -> None:
        f.write(struct.pack("<I", LIGHT_TYPE_NUM_AREA))
        super().write(f)
        f.write(self.size.to_bytes())

    @staticmethod
    def from_file(f: BinaryIO) -> "Light":
        size = glm.vec2.from_bytes(f.read(8))
        return AreaLight(size=size)


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
    def transform(self) -> mat4x4:
        rot = glm.inverse(glm.quatLookAt(self.direction, glm.vec3(0, -1, 0)))
        return glm.translate(self.translation) @ rot @ glm.scale(glm.vec3(self.radius))  # type:ignore[return-value]


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
    _voxel_texture: Texture3D | None = None
    global_id: int = field(default=0, compare=False)  # NOTE(david): This ID is volatile (doesn't need to be serialized)

    def __post_init__(self) -> None:
        global OBJECT_ID_COUNTER  # noqa: PLW0603
        self.global_id = OBJECT_ID_COUNTER
        if not self.name:
            self.name = f"{self.model.name}_{OBJECT_ID_COUNTER}"
        OBJECT_ID_COUNTER += 1
        super().__post_init__()
        self._center_translation: glm.vec3 = glm.vec3(0)
        self._center_translation.x = -(self.model.opengl_dimensions[0] // 2)
        self._center_translation.y = 0
        self._center_translation.z = -(self.model.opengl_dimensions[2] // 2)
        if self.model.opengl_dimensions[2] % 2 == 1:
            self._center_translation.z -= 1

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
    def gpu_transform_bytes(self) -> bytes:
        return b"".join(
            (
                self.transform.to_bytes(),
                glm.inverse(self.transform).to_bytes(),
                self.last_frame_transform.to_bytes(),
                glm.vec4(*self.model.opengl_dimensions, 1).to_bytes(),
            )
        )

    @property
    def center(self) -> glm.vec3:
        dim = glm.vec4(glm.vec3(self.model.opengl_dimensions) * 0.5, 1.0)  # type:ignore[call-overload]
        pos = self.transform * dim
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

            aabb_min = glm.min(aabb_min, world)  # type:ignore[assignment]
            aabb_max = glm.max(aabb_max, world)  # type:ignore[assignment]

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
    def transform(self) -> mat4x4:
        return (
            glm.translate(self.translation)  # type:ignore[return-value]
            @ glm.mat4_cast(self.rotation)
            @ glm.scale(self.scale)
            @ glm.translate(self._center_translation)
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
            _rotation=obj.rotation,
            _translation=obj.translation,
            _scale=obj.scale,
            model=model,
        )


class VoxelObjectGPUBuffer:
    TEXTURE_INSTANCE_SIZE = struct.calcsize("QII")
    TRANSFORM_INSTANCE_SIZE = struct.calcsize("16f16f16f4f")
    VISIBILITY_INSTANCE_SIZE = struct.calcsize("I")

    def __init__(self, ctx: Context, max_objects: int = MAX_VOXEL_OBJECTS) -> None:
        self.free_slots = deque(range(max_objects))
        self.slot_mapping: dict[int, int] = {}
        self.object_transform_buffer = ctx.buffer(
            reserve=self.TRANSFORM_INSTANCE_SIZE * MAX_VOXEL_OBJECTS,
            dynamic=True,
        )
        self.object_texture_buffer = ctx.buffer(
            reserve=self.TEXTURE_INSTANCE_SIZE * MAX_VOXEL_OBJECTS,
            dynamic=True,
        )
        self.visibility_buffer = ctx.buffer(
            reserve=self.VISIBILITY_INSTANCE_SIZE * MAX_VOXEL_OBJECTS,
            dynamic=True,
        )
        self.ctx = ctx

    def update_gpu_buffers(self, voxel_objects: list[VoxelObject], global_frame_counter: int) -> None:
        dirty_texture_slots: list[tuple[int, VoxelObject]] = []
        dirty_transform_slots: list[tuple[int, VoxelObject]] = []
        for obj in voxel_objects:
            if obj.global_id not in self.slot_mapping:
                if not self.free_slots:
                    msg = "No free slot for new object"
                    raise IndexError(msg)
                free_slot = self.free_slots.popleft()
                self.slot_mapping[obj.global_id] = free_slot
                dirty_texture_slots.append((free_slot, obj))

            # NOTE(david): last_frame_transform still needs to be updated one frame more
            #              could be optimized by just updating last_frame_transform for them
            if obj.last_frame_update + 1 < global_frame_counter:
                continue
            current_slot = self.slot_mapping[obj.global_id]
            dirty_transform_slots.append((current_slot, obj))
        dirty_texture_slots.sort(key=lambda x: x[0])
        dirty_transform_slots.sort(key=lambda x: x[0])

        for consecutive_region in self._group_into_consecutive_regions(dirty_texture_slots):
            texture_update_buffer = b"".join(
                struct.pack("<QII", obj.voxel_texture_handle, obj.model.palette_row, obj.model.material_row)
                for _, obj in consecutive_region
            )
            self.object_texture_buffer.write(
                texture_update_buffer,
                offset=consecutive_region[0][0] * self.TEXTURE_INSTANCE_SIZE,
            )

        for consecutive_region in self._group_into_consecutive_regions(dirty_transform_slots):
            transform_update_buffer = b"".join(obj.gpu_transform_bytes for _, obj in consecutive_region)
            self.object_transform_buffer.write(
                transform_update_buffer,
                offset=consecutive_region[0][0] * self.TRANSFORM_INSTANCE_SIZE,
            )

    def update_visibility_buffer(self, visible_objects: list[VoxelObject]) -> None:
        visible_slot_ids = b"".join(struct.pack("<I", self.slot_mapping[obj.global_id]) for obj in visible_objects)
        # TODO(david): can be replaced by array.array("i")
        self.visibility_buffer.write(visible_slot_ids, offset=0)

    @staticmethod
    def _group_into_consecutive_regions(
        sorted_slot_mapping: Sequence[tuple[int, VoxelObject]],
    ) -> Iterable[list[tuple[int, VoxelObject]]]:
        group: list[tuple[int, VoxelObject]] = []
        for el in sorted_slot_mapping:
            slot, _ = el
            if not group or slot != group[-1][0] + 1:
                if group:
                    yield group
                group = [el]
            else:
                group.append(el)
        if group:
            yield group


@dataclass(kw_only=True)
class Water(Object):
    color: glm.vec3 = field(default_factory=lambda: glm.vec3(1.0))
    vertices: list[glm.vec2] = field(default_factory=list)

    def upload_to_gpu(self) -> None:
        vao = VAO(self.name, mode=moderngl.TRIANGLE_FAN)
        # NOTE(david): counter z-fighting by moving down
        vertices_data = b"".join(glm.vec3(vert.x, -0.01, vert.y).to_bytes() for vert in self.vertices)
        vao.buffer(vertices_data, buffer_format="3f", attribute_names=["in_position"])
        self.geometry = vao

    @staticmethod
    def from_vox_water(vox_water: VoxWater) -> "Water":
        return Water(
            color=vox_water.color,
            vertices=vox_water.vertices,
            _translation=vox_water.translation,
            _rotation=vox_water.rotation,
        )

    def write(self, f: BinaryIO) -> None:
        super().write(f)
        f.write(self.color.to_bytes())
        f.write(struct.pack("<I", len(self.vertices)))
        f.writelines(vert.to_bytes() for vert in self.vertices)

    @staticmethod
    def from_file(f: BinaryIO) -> "Water":
        obj = Object.from_file(f)
        color = glm.vec3.from_bytes(f.read(12))
        vertices_num, *_ = struct.unpack("<I", f.read(4))
        vertices = [glm.vec2.from_bytes(f.read(8)) for _ in range(vertices_num)]
        return Water(
            name=obj.name,
            _translation=obj.translation,
            _rotation=obj.rotation,
            color=color,
            vertices=vertices,
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
        self.lights: list[Light] = []
        self.waters: list[Water] = []
        self.texture_information = TextureInformation()

    @staticmethod
    def from_vox_objects(vox_models: list[VoxModel], vox_lights: list[VoxLight], vox_waters: list[VoxWater]) -> "World":
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
                    _rotation=vox_model.rotation,
                    _translation=vox_model.translation,
                )
            )
        # TODO(david): support capsule lights
        supported_lights = [vox_light for vox_light in vox_lights if vox_light.light_type in ["sphere", "cone", "area"]]
        world.lights = [Light.from_vox_light(vox_light) for vox_light in supported_lights]
        world.waters = [Water.from_vox_water(vox_water) for vox_water in vox_waters]
        return world

    @staticmethod
    def from_file(world_file: Path) -> "World":
        world = World()
        with world_file.open("rb", buffering=1024 * 1024 * 10) as f:
            world.texture_information = TextureInformation.from_file(f)

            obj_num, *_ = struct.unpack("<I", f.read(4))
            for _ in range(obj_num):
                world.voxel_objects.append(VoxelObject.from_file(f, world.texture_information))

            obj_num, *_ = struct.unpack("<I", f.read(4))
            for _ in range(obj_num):
                world.lights.append(Light.from_file(f))

            obj_num, *_ = struct.unpack("<I", f.read(4))
            for _ in range(obj_num):
                world.waters.append(Water.from_file(f))

            assert f.read(1) == b""  # EOF
        return world

    def write(self, world_file: Path) -> None:
        with world_file.open("wb", buffering=1024 * 1024 * 10) as f:
            self.texture_information.write(f)

            f.write(struct.pack("<I", len(self.voxel_objects)))
            for vox_obj in self.voxel_objects:
                vox_obj.write(f)

            f.write(struct.pack("<I", len(self.lights)))
            for vox_light in self.lights:
                vox_light.write(f)

            f.write(struct.pack("<I", len(self.waters)))
            for vox_water in self.waters:
                vox_water.write(f)
