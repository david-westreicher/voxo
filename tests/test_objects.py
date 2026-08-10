import struct
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from moderngl import Context
from pyglm import glm

from voxo.model.level_parser import VoxLight, VoxWater
from voxo.model.model import SimplifiedModel
from voxo.model.vox_parser import MaterialType, VoxMaterial, VoxModel
from voxo.objects import AreaLight, VoxelObject, VoxelObjectGPUBuffer, World


def test_world_write_load(tmp_path: Path):
    # arrange
    vox_model = VoxModel(
        dimensions=(1, 2, 3),
        voxels=[(0, 0, 0, 1)],
        shape_name="test",
        palette=[(i, 0, 0) for i in range(256)],
        materials=[VoxMaterial(MaterialType.METAL)] * 256,
    )
    vox_light = VoxLight(name="test", light_type="sphere", size=glm.vec2(1, 2), light_size=0.5)
    vox_water = VoxWater(
        translation=glm.vec3(1.0, 2.0, 3.0),
        rotation=glm.quat(),
        color=glm.vec3(0.1, 0.2, 0.3),
        vertices=[glm.vec2(0, 0), glm.vec2(1, 1), glm.vec2(1, 0)],
    )
    world = World.from_vox_objects(vox_models=[vox_model], vox_lights=[vox_light], vox_waters=[vox_water])
    test_file = tmp_path / "test.level"

    # act
    world.write(test_file)
    parsed_world = World.from_file(test_file)

    # assert
    assert world.voxel_objects == parsed_world.voxel_objects
    assert world.lights == parsed_world.lights
    assert world.waters == parsed_world.waters
    assert world.texture_information == parsed_world.texture_information


@pytest.mark.parametrize(
    "area_position",
    [
        glm.vec2(0, 0),
        glm.vec2(1, 1),
        glm.vec2(-1, -1),
        glm.vec2(-1, 1),
        glm.vec2(1, -1),
    ],
)
def test_area_light_matrix(ctx: Context, area_position: glm.vec2):  # noqa: ARG001
    # arrange
    light = AreaLight(glm.vec2(1, 1))
    mat = light.area_light_matrix

    # act
    center_on_area = mat * glm.vec3(*area_position.to_tuple(), 1.0)

    # assert
    assert glm.epsilonEqual(center_on_area, glm.vec3(*area_position.to_tuple(), 0), 0.001)


def test_voxel_object_gpu_buffer_grouping():
    # arrange
    obj = Mock()
    sequence = [(0, obj), (1, obj), (2, obj), (5, obj), (6, obj), (8, obj), (9, obj), (10, obj), (12, obj)]

    # act
    groups = list(VoxelObjectGPUBuffer._group_into_consecutive_regions(sequence))

    # assert
    assert groups == [
        [(0, obj), (1, obj), (2, obj)],
        [(5, obj), (6, obj)],
        [(8, obj), (9, obj), (10, obj)],
        [(12, obj)],
    ]


def test_voxel_object_gpu_buffer_update_logic(ctx: Context):
    # arrange
    buffer = VoxelObjectGPUBuffer(ctx)
    model = SimplifiedModel("model", (0, 1, 1), b"", 1, 2)
    obj = VoxelObject(model=model, texture_information=Mock())
    obj.translation = glm.vec3(1, 2, 3)
    obj.scale = glm.vec3(4, 5, 6)
    obj.rotation = glm.angleAxis(glm.pi(), [7, 8, 9])
    obj.last_frame_update = 0
    obj.upload_to_gpu(ctx)

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, update texture + transform at timestep = 0
        buffer.update_gpu_buffers([obj], 0)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = [
            call(struct.pack("<QII", obj.voxel_texture_handle, 1, 2), offset=0),
            call(obj.gpu_transform_bytes, offset=0),
        ]
        assert actual_calls == expected_calls

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, only update the transform at timestep = 1
        buffer.update_gpu_buffers([obj], 1)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = [
            call(obj.gpu_transform_bytes, offset=0),
        ]
        assert actual_calls == expected_calls

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, no updates at timestep = 2
        buffer.update_gpu_buffers([obj], 2)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = []
        assert actual_calls == expected_calls

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, update transform timestep = 3
        obj.translation = glm.vec3(0, 0, 0)
        obj.last_frame_update = 3
        buffer.update_gpu_buffers([obj], 3)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = [
            call(obj.gpu_transform_bytes, offset=0),
        ]
        assert actual_calls == expected_calls

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, update transform for last_frame_transform timestep = 4
        buffer.update_gpu_buffers([obj], 4)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = [
            call(obj.gpu_transform_bytes, offset=0),
        ]
        assert actual_calls == expected_calls

    with patch("moderngl.Buffer.write") as mock_buffer_write:
        # act, no update at timestep = 5
        buffer.update_gpu_buffers([obj], 5)

        # assert
        actual_calls = mock_buffer_write.call_args_list
        expected_calls = []
        assert actual_calls == expected_calls


def test_voxel_object_gpu_buffer_update_2_objects(ctx: Context):
    # arrange
    buffer = VoxelObjectGPUBuffer(ctx)
    model_1 = SimplifiedModel("model", (0, 1, 1), b"", 1, 2)
    obj_1 = VoxelObject(model=model_1, texture_information=Mock())
    obj_1.voxel_texture_handle = 0
    model_2 = SimplifiedModel("model", (0, 1, 1), b"", 3, 4)
    obj_2 = VoxelObject(model=model_2, texture_information=Mock())
    obj_2.voxel_texture_handle = 1

    # act
    buffer.update_gpu_buffers([obj_1, obj_2], 0)

    # assert
    texture_data = buffer.object_texture_buffer.read(VoxelObjectGPUBuffer.TEXTURE_INSTANCE_SIZE * 2)
    assert texture_data == b"".join(
        (
            struct.pack("<QII", obj_1.voxel_texture_handle, 1, 2),
            struct.pack("<QII", obj_2.voxel_texture_handle, 3, 4),
        )
    )
    transform_data = buffer.object_transform_buffer.read(VoxelObjectGPUBuffer.TRANSFORM_INSTANCE_SIZE * 2)
    assert transform_data == b"".join((obj_1.gpu_transform_bytes, obj_2.gpu_transform_bytes))
