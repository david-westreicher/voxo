import struct
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np

VoxelInfo = tuple[int, int, int, int]  # x, y, z, color index


@dataclass
class SimplifiedModel:
    name: str
    opengl_dimensions: tuple[int, int, int]
    voxel_data: bytes = field(repr=False)
    palette_row: int
    material_row: int

    def write(self, f: BinaryIO) -> None:
        f.write(struct.pack("<III", *self.opengl_dimensions))
        f.write(self.voxel_data)
        f.write(struct.pack("<I", self.palette_row))
        f.write(struct.pack("<I", self.material_row))

    @staticmethod
    def from_file(f: BinaryIO, model_name: str) -> "SimplifiedModel":
        (w, h, d) = struct.unpack("<III", f.read(12))
        voxel_data = f.read(w * h * d)
        palette_row, *_ = struct.unpack("<I", f.read(4))
        material_row, *_ = struct.unpack("<I", f.read(4))
        return SimplifiedModel(
            name=model_name,
            opengl_dimensions=(w, h, d),
            voxel_data=voxel_data,
            palette_row=palette_row,
            material_row=material_row,
        )


@dataclass
class Model:
    name: str
    opengl_dimensions: tuple[int, int, int]
    voxel_data: bytes
    palette_data: bytes
    material_data: bytes

    def serialize(self, f: BinaryIO) -> None:
        f.write(struct.pack("<III", *self.opengl_dimensions))
        f.write(self.voxel_data)
        palette_data = self.palette_data
        f.write(struct.pack("<I", len(palette_data)))
        f.write(palette_data)
        material_data = self.material_data
        f.write(struct.pack("<I", len(material_data)))
        f.write(material_data)

    @staticmethod
    def deserialize(f: BinaryIO, model_name: str) -> "Model":
        (w, h, d) = struct.unpack("<III", f.read(12))
        voxel_data = f.read(w * h * d)
        palette_data_len, *_ = struct.unpack("<I", f.read(4))
        palette_data = f.read(palette_data_len)
        material_data_len, *_ = struct.unpack("<I", f.read(4))
        material_data = f.read(material_data_len)
        return Model(
            name=model_name,
            opengl_dimensions=(w, h, d),
            voxel_data=voxel_data,
            palette_data=palette_data,
            material_data=material_data,
        )

    def simplify(self, palette_row: int, material_row: int) -> SimplifiedModel:
        return SimplifiedModel(
            self.name,
            self.opengl_dimensions,
            self.voxel_data,
            palette_row=palette_row,
            material_row=material_row,
        )


@dataclass
class Material:
    reflectivity: float = 0.0
    roughness: float = 0.0
    metallic: float = 0.0
    emissive: float = 0.0
    transparency: float = 0.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.reflectivity,
            self.roughness,
            self.metallic,
            self.emissive or -self.transparency or 0.0,
        )


def generate_model(
    name: str,
    dimensions: tuple[int, int, int],
    voxels: list[VoxelInfo],
    palette: list[tuple[int, int, int]],
    materials: list[Material],
) -> Model:
    assert len(palette) == len(materials)
    used_palette_indices = sorted({col for _, _, _, col in voxels})
    assert 0 not in used_palette_indices
    palette_indirection_map = {
        original_palette_index: i + 1 for i, original_palette_index in enumerate(used_palette_indices)
    }
    palette_indirection_map[0] = 0

    voxel_data = []
    voxel_map = {(x, y, z): col for x, y, z, col in voxels}
    w, h, d = dimensions
    for y in reversed(range(h)):
        for z in range(d):
            for x in range(w):
                assert 0 <= x < w
                assert 0 <= y < h
                assert 0 <= z < d
                col = palette_indirection_map[voxel_map.get((x, y, z), 0)]
                voxel_data.append(col)

    palette_data = [0] * 3
    material_data = [0.0] * 4
    for i in used_palette_indices:
        palette_data.extend(palette[i - 1])
        mat = materials[i - 1]
        material_data.extend(mat.to_tuple())

    opengl_dimensions = (w, d, h)
    return Model(
        name=name,
        opengl_dimensions=opengl_dimensions,
        voxel_data=bytes(voxel_data),
        palette_data=bytes(palette_data),
        material_data=np.array(material_data, dtype=np.float16).tobytes(),
    )
