from pathlib import Path

import numpy as np

from voxo.model import parse_vox_file
from voxo.model.model import Material, generate_model

UNUSED_COLOR = (0, 0, 0)
USED_COLOR_0 = (255, 0, 0)
USED_COLOR_1 = (0, 255, 0)
USED_COLOR_2 = (0, 0, 255)
UNUSED_MATERIAL = Material(0.0, 0.0, 0.0, 0.0)
USED_MATERIAL_0 = Material(1.0, 0.0, 0.0, 0.0)
USED_MATERIAL_1 = Material(0.0, 1.0, 0.0, 0.0)
USED_MATERIAL_2 = Material(0.0, 0.0, 1.0, 0.0)


def test_generate_model_palette_compression():
    # arrange
    dimensions = (1, 1, 3)
    palette = [
        UNUSED_COLOR,
        UNUSED_COLOR,
        UNUSED_COLOR,
        USED_COLOR_0,
        UNUSED_COLOR,
        USED_COLOR_1,
        USED_COLOR_2,
    ]
    materials = [
        UNUSED_MATERIAL,
        UNUSED_MATERIAL,
        UNUSED_MATERIAL,
        USED_MATERIAL_0,
        UNUSED_MATERIAL,
        USED_MATERIAL_1,
        USED_MATERIAL_2,
    ]
    voxels = [(0, 0, 0, 4), (0, 0, 1, 6), (0, 0, 2, 7)]

    # act
    model = generate_model("", dimensions, voxels, palette, materials)

    # assert
    expected_material_data = np.array(
        (0.0,) * 4 + USED_MATERIAL_0.to_tuple() + USED_MATERIAL_1.to_tuple() + USED_MATERIAL_2.to_tuple(),
        dtype=np.float16,
    ).tobytes()
    expected_voxel_data = [1, 2, 3]
    assert model.palette_data == b"\x00\x00\x00" + b"\xff\x00\x00" + b"\x00\xff\x00" + b"\x00\x00\xff"
    assert model.material_data == expected_material_data
    assert model.voxel_data == bytes(expected_voxel_data)


def test_generate_model_palette_compression_vox_file():
    # arrange
    model = parse_vox_file(Path("./tests/test_data/simple.vox")).get_model("simple_gizmo").to_model()

    # assert
    assert model.palette_data == b"\x00\x00\x00" + b"\xff\x00\x00" + b"\x00\xff\x00" + b"\x00\x00\xff" + b"\xff\xff\xff"


def test_generate_model_material_compression_vox_file():
    # arrange
    model = parse_vox_file(Path("./tests/test_data/materials.vox")).get_model("simple").to_model()

    # assert
    expected_materials = [
        Material(),
        Material(roughness=0.1),
        Material(roughness=0.304, metallic=1.0),
        Material(roughness=0.1, emissive=1.0),
        Material(roughness=0.1, transparency=0.5),
    ]
    assert model.palette_data == b"\x00\x00\x00" + b"\xff\xff\xff" + b"\xff\x00\x00" + b"\x00\xff\x00" + b"\x00\x00\xff"
    assert (
        model.material_data
        == np.array([e for mat in expected_materials for e in mat.to_tuple()], dtype=np.float16).tobytes()
    )
