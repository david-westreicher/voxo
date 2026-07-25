from dataclasses import dataclass
from pathlib import Path

from .model import Model, VoxelInfo, generate_palette_data, generate_voxel_data


@dataclass
class TextModel:
    path: Path
    voxels: list[VoxelInfo]
    dimensions: tuple[int, int, int] = (0, 0, 0)

    @property
    def opengl_dimensions(self) -> tuple[int, int, int]:
        (w, h, d) = self.dimensions
        return (w, d, h)

    def __post_init__(self) -> None:
        (min_x, min_y, min_z), _ = self.get_min_max(self.voxels)
        self.voxels = [(x - min_x, y - min_y, z - min_z, color_index) for x, y, z, color_index in self.voxels]
        _, (w, h, d) = self.get_min_max(self.voxels)
        self.dimensions = (w + 1, h + 1, d + 1)

    def get_min_max(self, voxels: list[VoxelInfo]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        min_x = min(x for x, _, _, _ in voxels)
        max_x = max(x for x, _, _, _ in voxels)
        min_y = min(y for _, y, _, _ in voxels)
        max_y = max(y for _, y, _, _ in voxels)
        min_z = min(z for _, _, z, _ in voxels)
        max_z = max(z for _, _, z, _ in voxels)
        return (min_x, min_y, min_z), (max_x, max_y, max_z)


def convert_hex_to_rgb(hex_col: int) -> tuple[int, int, int]:
    return ((hex_col >> 16) & 0xFF, (hex_col >> 8) & 0xFF, hex_col & 0xFF)


def parse_text_model(model_path: Path) -> Model:
    voxels: list[VoxelInfo] = []
    with model_path.open("r") as f:
        for line in f:
            if line.startswith("# "):
                continue
            x, y, z, col = [int(e) if i < 3 else int(e, base=16) for i, e in enumerate(line.strip().split())]
            voxels.append((x, y, z, col))
    hex_palette = sorted({col for _, _, _, col in voxels})
    palette = sorted(convert_hex_to_rgb(col) for col in hex_palette)
    voxels = [(x, y, z, hex_palette.index(col) + 1) for x, y, z, col in voxels]
    text_model = TextModel(path=model_path, voxels=voxels)
    return Model(
        model_path.with_suffix("").name,
        text_model.opengl_dimensions,
        generate_voxel_data(text_model.dimensions, text_model.voxels),
        generate_palette_data(palette),
    )
