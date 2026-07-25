import struct
from dataclasses import dataclass
from typing import BinaryIO

VoxelInfo = tuple[int, int, int, int]  # x, y, z, color index


@dataclass
class Model:
    name: str
    opengl_dimensions: tuple[int, int, int]
    voxel_data: bytes
    palette_data: bytes

    def serialize(self, f: BinaryIO) -> None:
        f.write(struct.pack("<III", *self.opengl_dimensions))
        f.write(self.voxel_data)
        palette_data = self.palette_data
        f.write(struct.pack("<I", len(palette_data)))
        f.write(palette_data)

    @staticmethod
    def deserialize(f: BinaryIO, model_name: str) -> "Model":
        (w, h, d) = struct.unpack("<III", f.read(12))
        voxel_data = f.read(w * h * d)
        palette_data_len, *_ = struct.unpack("<I", f.read(4))
        palette_data = f.read(palette_data_len)
        return Model(
            name=model_name,
            opengl_dimensions=(w, h, d),
            voxel_data=voxel_data,
            palette_data=palette_data,
        )


def generate_voxel_data(dimensions: tuple[int, int, int], voxels: list[VoxelInfo]) -> bytes:
    voxel_map = {(x, y, z): col for x, y, z, col in voxels}
    voxel_data = []
    max_x, max_y, max_z = dimensions
    for y in reversed(range(max_y)):
        for z in range(max_z):
            for x in range(max_x):
                col = voxel_map.get((x, y, z), 0)
                voxel_data.append(col)
    return bytes(voxel_data)


def generate_palette_data(palette: list[tuple[int, int, int]]) -> bytes:
    palette_data = [0] * 3
    for r, g, b in palette:
        palette_data.extend([r, g, b])
    return bytes(palette_data)
