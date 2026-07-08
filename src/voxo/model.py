from dataclasses import dataclass
from pathlib import Path

VoxelInfo = tuple[int, int, int, int]  # x, y, z, color index


@dataclass
class Model:
    palette: list[tuple[int, int, int]]
    voxels: list[VoxelInfo]
    dimensions: tuple[int, int, int] = (0, 0, 0)

    @property
    def opengl_dimensions(self) -> tuple[int, int, int]:
        (w, h, d) = self.dimensions
        return (w, d, h)

    def __post_init__(self) -> None:
        assert len(self.palette) <= 256, "Palette can have at most 256 colors"  # noqa: PLR2004
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

    def generate_voxel_data(self) -> bytes:
        voxel_map = {(x, y, z): col + 1 for x, y, z, col in self.voxels}
        voxel_data = []
        max_x, max_y, max_z = self.dimensions
        for y in reversed(range(max_y)):
            for z in range(max_z):
                for x in range(max_x):
                    col = voxel_map.get((x, y, z), 0)
                    voxel_data.append(col)
        return bytes(voxel_data)

    def generate_palette_data(self) -> bytes:
        print(self.palette, len(self.palette))
        palette_data = [0] * 3
        for r, g, b in self.palette:
            palette_data.extend([r, g, b])
        return bytes(palette_data)

    def serialize(self, model_path: Path) -> None:
        with model_path.open("wb") as f:
            for dim in self.dimensions:
                f.write(dim.to_bytes(4, "big"))
            f.write(len(self.palette).to_bytes(1, "big"))
            for col in self.palette:
                for channel in col:
                    f.write(channel.to_bytes(1, "big"))
            f.write(self.generate_voxel_data())


def convert_hex_to_rgb(hex_col: int) -> tuple[int, int, int]:
    return ((hex_col >> 16) & 0xFF, (hex_col >> 8) & 0xFF, hex_col & 0xFF)


def parse_model(model_path: Path) -> Model:
    voxels: list[VoxelInfo] = []
    with model_path.open("r") as f:
        for line in f:
            if line.startswith("# "):
                continue
            x, y, z, col = [int(e) if i < 3 else int(e, base=16) for i, e in enumerate(line.strip().split())]  # noqa: PLR2004
            voxels.append((x, y, z, col))
    hex_palette = sorted({col for _, _, _, col in voxels})
    palette = sorted(convert_hex_to_rgb(col) for col in hex_palette)
    voxels = [(x, y, z, hex_palette.index(col)) for x, y, z, col in voxels]
    return Model(palette=palette, voxels=voxels)


if __name__ == "__main__":
    model = parse_model(Path("./resources/models/chr_rain.txt"))
    print(model, model.generate_voxel_data())
