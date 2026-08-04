from pathlib import Path

import pytest
from pyglm import glm

from voxo.model.vox_parser import MaterialType, VoxMaterial, parse_vox_file


def test_simple_voxel_file_with_one_model():
    # act
    simplified_voxel_file = parse_vox_file(Path("./tests/test_data/simple.vox"))

    # assert
    assert simplified_voxel_file.shape_names == ["simple_gizmo"]
    model = simplified_voxel_file.get_model("simple_gizmo")
    assert model.shape_name == "simple_gizmo"
    assert model.opengl_dimensions == (3, 3, 3)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(0, 0, 1)
    assert model.palette == [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)] + [(75, 75, 75)] * 251 + [
        (0, 0, 0)
    ]
    assert sorted(model.voxels) == sorted(
        [(0, 0, 0, 4), (1, 0, 0, 1), (2, 0, 0, 1), (0, 1, 0, 2), (0, 2, 0, 2), (0, 0, 1, 3), (0, 0, 2, 3)]
    )


def test_simple_voxel_file_with_rotation_raises():
    # act / assert
    with pytest.raises(AssertionError):
        parse_vox_file(Path("./tests/test_data/simple_rotation.vox"))


def test_grouped_models():
    # act
    simplified_voxel_file = parse_vox_file(Path("./tests/test_data/group.vox"))

    # assert
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)] + [(75, 75, 75)] * 251 + [(0, 0, 0)]
    assert sorted(simplified_voxel_file.shape_names) == sorted(["obj_0", "obj_1", "obj_2", "obj_3"])

    model = simplified_voxel_file.get_model("obj_0")
    assert model.shape_name == "obj_0"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, -2, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 4)])

    model = simplified_voxel_file.get_model("obj_1")
    assert model.shape_name == "obj_1"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(1, -2, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 1)])

    model = simplified_voxel_file.get_model("obj_2")
    assert model.shape_name == "obj_2"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, 1, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 2)])

    model = simplified_voxel_file.get_model("obj_3")
    assert model.shape_name == "obj_3"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, -2, 1)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 3)])


def test_grouped_models_transform_is_ignored():
    # act
    simplified_voxel_file = parse_vox_file(Path("./tests/test_data/group_transform.vox"))

    # assert
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)] + [(75, 75, 75)] * 251 + [(0, 0, 0)]
    assert sorted(simplified_voxel_file.shape_names) == sorted(["obj_0", "obj_1", "obj_2", "obj_3"])

    model = simplified_voxel_file.get_model("obj_0")
    assert model.shape_name == "obj_0"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, -2, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 4)])

    model = simplified_voxel_file.get_model("obj_1")
    assert model.shape_name == "obj_1"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(1, -2, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 1)])

    model = simplified_voxel_file.get_model("obj_2")
    assert model.shape_name == "obj_2"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, 1, -2)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 2)])

    model = simplified_voxel_file.get_model("obj_3")
    assert model.shape_name == "obj_3"
    assert model.palette == palette
    assert model.opengl_dimensions == (1, 1, 1)
    assert model.rotation == glm.quat()
    assert model.translation == glm.vec3(-2, -2, 1)
    assert sorted(model.voxels) == sorted([(0, 0, 0, 3)])


def test_dimensions():
    # act
    simplified_voxel_file = parse_vox_file(Path("./tests/test_data/simple_dimensions.vox"))

    # assert
    assert sorted(simplified_voxel_file.shape_names) == sorted(["simple"])

    model = simplified_voxel_file.get_model("simple")
    assert model.opengl_dimensions == (1, 3, 2)
    assert model.shape_name == "simple"
    assert sorted(model.voxels) == sorted(
        [
            (0, 0, 0, 4),
            (0, 0, 1, 4),
            (0, 1, 0, 4),
            (0, 1, 1, 4),
            (0, 0, 2, 4),
            (0, 1, 2, 4),
        ]
    )


def test_material():
    # act
    simplified_voxel_file = parse_vox_file(Path("./tests/test_data/materials.vox"))

    # assert
    assert sorted(simplified_voxel_file.shape_names) == sorted(["simple"])

    model = simplified_voxel_file.get_model("simple")
    assert model.shape_name == "simple"
    assert sorted(model.voxels) == sorted(
        [
            (0, 0, 0, 2),
            (0, 0, 1, 4),
            (0, 1, 0, 3),
            (0, 1, 1, 1),
        ]
    )
    assert model.palette[:4] == [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert model.materials[:4] == [
        VoxMaterial(
            material_type=MaterialType.DIFFUSE,
            blend_fresnel=0.0,
            refractive_index=0.0,
            density=0.0,
            roughness=0.0,
        ),
        VoxMaterial(
            material_type=MaterialType.METAL,
            blend_fresnel=0.3,
            refractive_index=0.0,
            density=0.00,
            roughness=0.304044,
            metallic=1.0,
        ),
        VoxMaterial(
            material_type=MaterialType.EMIT,
            blend_fresnel=0.0,
            refractive_index=0.0,
            density=0.0,
            roughness=0.0,
            emission=1.0,
            flux=1.0,
            low_dynamic_range_intensity=0.3,
        ),
        VoxMaterial(
            material_type=MaterialType.GLASS,
            blend_fresnel=0.3,
            refractive_index=1.3,
            density=0.05,
            roughness=0.1,
            media_type="_scatter",
            alpha=0.5,
            phase=-0.04,
        ),
    ]
