from pathlib import Path

import pytest
from moderngl import Context
from pyglm import glm

from voxo.model.level_parser import VoxLight
from voxo.model.vox_parser import MaterialType, VoxMaterial, VoxModel
from voxo.objects import AreaLight, World


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
    world = World.from_vox_objects(vox_models=[vox_model], vox_lights=[vox_light])
    test_file = tmp_path / "test.level"

    # act
    world.write(test_file)
    parsed_world = World.from_file(test_file)

    # assert
    assert world.voxel_objects == parsed_world.voxel_objects
    assert world.lights == parsed_world.lights
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
