import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO, cast

from pyglm import glm

from .model import Material as ModelMaterial
from .model import Model, VoxelInfo, generate_model


@dataclass
class ModelContainer:
    model_id: int
    attributes: dict[str, str]


@dataclass
class Size:
    x: int
    y: int
    z: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass
class Colors:
    items: list[Color]

    def to_tuples(self) -> list[tuple[int, int, int]]:
        return [col.to_tuple() for col in self.items]


@dataclass
class Voxel:
    x: int
    y: int
    z: int
    color_index: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.z, self.color_index)


@dataclass
class Voxels:
    items: list[Voxel]

    def to_voxel_info(self) -> list[VoxelInfo]:
        return [vox.to_tuple() for vox in self.items]


@dataclass
class Material:
    material_id: int
    attributes: dict[str, str]


@dataclass
class IndirectionMap:
    indices: list[int]


@dataclass
class Note:
    color_names: list[str]


@dataclass
class Meta:
    attributes: dict[str, str]


@dataclass
class Layer:
    layer_id: int
    attributes: dict[str, str]


@dataclass
class Camera:
    camera_id: int
    attributes: dict[str, str]


@dataclass
class RenderObject:
    attributes: dict[str, str]


@dataclass
class Transform:
    node_id: int
    attributes: dict[str, str]
    child_node_id: int
    reserved_id: int
    layer_id: int
    frames: list[dict[str, str]]


@dataclass
class Group:
    node_id: int
    attributes: dict[str, str]
    children: list[int]


@dataclass
class Shape:
    node_id: int
    attributes: dict[str, str]
    models: list[ModelContainer]


def parse_string(f: BinaryIO) -> str:
    length, *_ = struct.unpack("<i", f.read(4))
    value, *_ = struct.unpack(f"<{length}s", f.read(length))
    assert type(value) is bytes
    return value.decode()


def parse_dict(f: BinaryIO) -> dict[str, str]:
    kv_pairs, *_ = struct.unpack("<i", f.read(4))
    dictionary = {}
    for _ in range(kv_pairs):
        key = parse_string(f)
        value = parse_string(f)
        dictionary[key] = value
    return dictionary


def parse_frame(f: BinaryIO) -> dict[str, str]:
    return parse_dict(f)


def parse_model(f: BinaryIO) -> ModelContainer:
    model_id, *_ = struct.unpack("<i", f.read(4))
    attributes = parse_dict(f)
    return ModelContainer(model_id, attributes)


ChunkContent = (
    Size
    | Voxels
    | Transform
    | Group
    | Shape
    | Colors
    | Note
    | Material
    | IndirectionMap
    | Meta
    | Layer
    | RenderObject
    | Camera
)


def parse_chunk_content(f: BinaryIO, chunk_type: str, n: int) -> ChunkContent:  # noqa: C901, PLR0911, PLR0912, PLR0915
    if chunk_type == "SIZE":
        assert n == 12
        x, y, z = struct.unpack("<III", f.read(12))
        return Size(x, y, z)
    if chunk_type == "XYZI":
        num_voxels, *_ = struct.unpack("<I", f.read(4))
        assert n == num_voxels * 4 + 4
        voxel_data = struct.unpack(f">{num_voxels}I", f.read(num_voxels * 4))
        voxels = []
        for i in voxel_data:
            x = (i >> 24) & 0xFF
            y = (i >> 16) & 0xFF
            z = (i >> 8) & 0xFF
            color_index = i & 0xFF
            voxels.append(Voxel(x, y, z, color_index))
        return Voxels(voxels)
    if chunk_type == "nTRN":
        node_id, *_ = struct.unpack("<I", f.read(4))
        transform_attributes = parse_dict(f)
        child_node_id, reserved_id, layer_id, num_of_frames = struct.unpack("<iiii", f.read(16))
        assert reserved_id == -1
        assert num_of_frames > 0
        assert layer_id in [0, -1]
        assert num_of_frames == 1
        # NOTE(david): layer and num_of_frames not needed for this use case
        frames = [parse_dict(f) for _ in range(num_of_frames)]
        return Transform(node_id, transform_attributes, child_node_id, reserved_id, layer_id, frames)
    if chunk_type == "nGRP":
        node_id, *_ = struct.unpack("<I", f.read(4))
        group_attributes = parse_dict(f)
        num_child_nodes, *_ = struct.unpack("<I", f.read(4))
        children = []
        for _ in range(num_child_nodes):
            child_node_id, *_ = struct.unpack("<I", f.read(4))
            children.append(child_node_id)
        return Group(node_id, group_attributes, children)
    if chunk_type == "nSHP":
        node_id, *_ = struct.unpack("<I", f.read(4))
        shape_attributes = parse_dict(f)
        num_models, *_ = struct.unpack("<I", f.read(4))
        models = [parse_model(f) for _ in range(num_models)]
        return Shape(node_id, shape_attributes, models)
    if chunk_type == "RGBA":
        palette_data = struct.unpack(">256I", f.read(256 * 4))
        palette = []
        for val in palette_data:
            r = (val >> 24) & 0xFF
            g = (val >> 16) & 0xFF
            b = (val >> 8) & 0xFF
            a = (val >> 0) & 0xFF
            palette.append(Color(r, g, b, a))
        return Colors(palette)
    if chunk_type == "NOTE":
        num_col_names, *_ = struct.unpack("<I", f.read(4))
        color_names = [parse_string(f) for _ in range(num_col_names)]
        return Note(color_names)
    if chunk_type == "MATL":
        material_id, *_ = struct.unpack("<I", f.read(4))
        material_attributes = parse_dict(f)
        return Material(material_id, material_attributes)
    if chunk_type == "IMAP":
        indices = []
        for _ in range(256):
            palette_index_association, *_ = struct.unpack("<B", f.read(1))
            indices.append(palette_index_association)
        return IndirectionMap(indices)
    if chunk_type == "META":
        attributes = parse_dict(f)
        return Meta(attributes)
    if chunk_type == "LAYR":
        layer_id, *_ = struct.unpack("<I", f.read(4))
        layer_attributes = parse_dict(f)
        reserved_id, *_ = struct.unpack("<i", f.read(4))
        assert reserved_id == -1
        return Layer(layer_id, layer_attributes)
    if chunk_type == "rOBJ":
        attributes = parse_dict(f)
        return RenderObject(attributes)
    if chunk_type == "rCAM":
        camera_id, *_ = struct.unpack("<I", f.read(4))
        attributes = parse_dict(f)
        return Camera(camera_id, attributes)
    raise NotImplementedError(chunk_type)


def parse_chunk(f: BinaryIO, parsed_chunk_buffer: list[ChunkContent]) -> int:
    (chunk_type, n, m) = struct.unpack("<4sII", f.read(12))
    parsed_bytes = 12
    if n > 0:
        chunk_content = parse_chunk_content(f, chunk_type.decode(), n)
        parsed_chunk_buffer.append(chunk_content)
        parsed_bytes += n
    while m > 0:
        m -= parse_chunk(f, parsed_chunk_buffer)
    assert m == 0
    return parsed_bytes


class MaterialType(Enum):
    DIFFUSE = ""
    METAL = "_metal"
    EMIT = "_emit"
    GLASS = "_glass"


@dataclass
class VoxMaterial:
    material_type: MaterialType = MaterialType.DIFFUSE
    blend_fresnel: float = 0.0
    refractive_index: float = 0.0
    density: float = 0.0

    # metallic
    metallic: float = 0.0
    roughness: float = 0.0
    specular_power: float = 0.0

    # emitting
    emission: float = 0.0
    flux: float = 0.0
    low_dynamic_range_intensity: float = 0.0

    # glass
    media_type: str = ""
    alpha: float = 0.0
    phase: float = 0.0

    @staticmethod
    def from_material(attributes: dict[str, str]) -> "VoxMaterial":
        mat = VoxMaterial()
        mat.material_type = cast("MaterialType", MaterialType._value2member_map_[attributes.get("_type", "")])
        if mat.material_type == MaterialType.DIFFUSE:
            pass
        elif mat.material_type == MaterialType.METAL:
            mat.roughness = max(0.0, float(attributes.get("_rough", 0.0)))
            mat.blend_fresnel = float(attributes.get("_ior", 0.0))
            mat.specular_power = float(attributes.get("_sp", 0.0))
            mat.metallic = float(attributes.get("_metal", 0.0))
        elif mat.material_type == MaterialType.EMIT:
            mat.emission = float(attributes.get("_emit", 0.0))
            mat.flux = float(attributes.get("_flux", 0.0))
            mat.low_dynamic_range_intensity = float(attributes.get("_ldr", 0.0))
        elif mat.material_type == MaterialType.GLASS:
            mat.roughness = max(0.0, float(attributes.get("_rough", 0.0)))
            mat.blend_fresnel = float(attributes.get("_ior", 0.0))
            mat.alpha = max(float(attributes.get("_alpha", 0.0)), float(attributes.get("_trans", 0.0)))
            mat.refractive_index = float(attributes.get("_ri", 0.0))
            mat.media_type = attributes.get("_media_type", "")
            mat.density = float(attributes.get("_d", 0.0))
            mat.phase = float(attributes.get("_g", 0.0))
        else:
            raise NotImplementedError
        return mat


@dataclass
class VoxModel:
    dimensions: tuple[int, int, int]
    voxels: list[VoxelInfo]
    shape_name: str = ""
    translation: glm.vec3 = field(default_factory=lambda: glm.vec3(0))
    rotation: glm.quat = field(default_factory=glm.quat)
    palette: list[tuple[int, int, int]] = field(default_factory=list)
    materials: list[VoxMaterial] = field(default_factory=list)

    @property
    def opengl_dimensions(self) -> tuple[int, int, int]:
        (w, h, d) = self.dimensions
        return (w, d, h)

    def __repr__(self) -> str:
        return (
            f"VoxModel(shape_name={self.shape_name}, dimensions={self.opengl_dimensions}, voxels={self.voxels[:4]}, "
            f"palette={self.palette[:4]}, translation={self.translation})"
        )

    def to_model(self) -> Model:
        assert len(self.palette) == len(self.materials) == 256
        converted_materials = []
        for mat in self.materials:
            conv = (
                ModelMaterial(roughness=1.0)
                if mat.material_type == MaterialType.DIFFUSE
                else ModelMaterial(
                    reflectivity=max(0.0, mat.specular_power - 1.0),
                    roughness=mat.roughness,
                    metallic=mat.metallic,  # or (1.0 if mat.material_type == MaterialType.METAL else 0.0),
                    emissive=mat.emission * glm.pow(10, mat.flux - 1),
                    transparency=mat.alpha,
                )
            )
            converted_materials.append(conv)
            assert 0.0 <= conv.reflectivity <= 1.0, conv.reflectivity
            assert 0.0 <= conv.roughness <= 1.0, conv.roughness
            assert 0.0 <= conv.metallic <= 1.0, conv.metallic
            assert 0.0 <= conv.emissive <= 10_000.0, conv.emissive
        return generate_model(
            name=self.shape_name,
            voxels=self.voxels,
            palette=self.palette,
            materials=converted_materials,
        )

    def copy(self) -> "VoxModel":
        return VoxModel(
            dimensions=self.dimensions,
            voxels=self.voxels,
            shape_name=self.shape_name + "_",
            palette=self.palette,
            materials=self.materials,
        )


def generate_vox_models(chunk_buffer: list[ChunkContent]) -> dict[str, VoxModel]:  # noqa: C901
    current_size = Size(0, 0, 0)
    models: list[VoxModel] = []
    shape_by_node_id: dict[int, Shape] = {}
    for item in chunk_buffer:
        if type(item) is not Shape:
            continue
        shape_by_node_id[item.node_id] = item

    palette: Colors | None = None
    materials: dict[int, VoxMaterial] = {}
    for item in chunk_buffer:
        if type(item) is Size:
            current_size = item
        if type(item) is Voxels:
            models.append(VoxModel(dimensions=current_size.to_tuple(), voxels=item.to_voxel_info()))
        if type(item) is Transform:
            if "_name" not in item.attributes:
                # could be a nested scene graph
                continue
            assert len(item.frames) == 1
            assert "_r" not in item.frames[0]
            if "_t" in item.frames[0]:
                translation = [float(e) for e in item.frames[0]["_t"].split()]
            else:
                translation = [0.0, 0.0, 0.0]
            shape_name = item.attributes["_name"]
            shape = shape_by_node_id[item.child_node_id]
            assert len(shape.models) == 1
            model_id = shape.models[0].model_id
            assert not models[model_id].shape_name
            models[model_id].shape_name = shape_name
            models[model_id].translation = glm.vec3(translation)
        if type(item) is Material:
            materials[item.material_id - 1] = VoxMaterial.from_material(item.attributes)
        if type(item) is Colors:
            assert not palette
            palette = item
    assert palette

    shape_map = {}
    for model in models:
        model.palette = palette.to_tuples()
        model.materials = [materials[i] if i in materials else VoxMaterial() for i in range(256)]
        shape_map[model.shape_name] = model
    return shape_map


class SimplifiedVoxFile:
    def __init__(self, path: Path, shape_to_obj_map: dict[str, VoxModel]) -> None:
        self.shape_to_obj_map = shape_to_obj_map
        self.path = path

    def get_model(self, shape: str) -> VoxModel:
        return self.shape_to_obj_map[shape]

    @property
    def shape_names(self) -> list[str]:
        return list(self.shape_to_obj_map)

    @staticmethod
    def from_file(vox_file: Path) -> "SimplifiedVoxFile":
        with vox_file.open("rb", buffering=10 * 1024 * 1024) as f:  # read with 10MB buffer
            magick, version_number = struct.unpack("<4sI", f.read(8))
            assert magick.decode() == "VOX ", magick
            assert version_number == 200, version_number
            chunk_buffer: list[ChunkContent] = []
            parse_chunk(f, chunk_buffer)
            assert f.read(1) == b""  # we should be at end of file

        vox_objects = generate_vox_models(chunk_buffer)
        return SimplifiedVoxFile(vox_file, vox_objects)


def parse_vox_file(vox_file: Path) -> SimplifiedVoxFile:
    return SimplifiedVoxFile.from_file(vox_file)
