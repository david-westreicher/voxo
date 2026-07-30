from pathlib import Path

from voxo.model.vox_parser import MaterialType, VoxMaterial, VoxModel
from voxo.objects import World


def test_world_write_load(tmp_path: Path):
    # arrange
    vox_model = VoxModel(
        dimensions=(1, 2, 3),
        voxels=[(0, 0, 0, 1)],
        shape_name="test",
        palette=[(i, 0, 0) for i in range(256)],
        materials=[VoxMaterial(MaterialType.METAL)] * 256,
    )
    world = World.from_vox_models([vox_model])
    test_file = tmp_path / "test.level"

    # act
    world.write(test_file)
    parsed_world = World.from_file(test_file)

    # assert
    assert world.voxel_objects == parsed_world.voxel_objects
    assert world.texture_information == parsed_world.texture_information
