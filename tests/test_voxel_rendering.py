import numpy as np
import pytest
from moderngl import Context
from moderngl_window.context.base.window import WindowConfig
from pyglm import glm

from voxo.model.model import Material, generate_model
from voxo.objects import TextureInformation, VoxelObject
from voxo.voxel_rendering import GlobalOccluder


def create_cube(dimensions: tuple[int, int, int], ctx: Context) -> VoxelObject:
    vox_obj = VoxelObject(
        name="",
        texture_information=TextureInformation(),
        model=generate_model(
            name="",
            dimensions=dimensions,
            voxels=[
                (x, y, z, 1) for x in range(dimensions[0]) for y in range(dimensions[1]) for z in range(dimensions[2])
            ],
            palette=[(0, 0, 0)],
            materials=[Material()],
        ),
    )
    vox_obj.upload_to_gpu(ctx)
    return vox_obj


def fill_occluder(global_occluder: GlobalOccluder, ctx: Context) -> None:
    vox_obj = create_cube((6, 6, 6), ctx)
    vox_obj.translation = glm.vec3(3, 0, 3)
    global_occluder.blit_object(vox_obj)
    ctx.finish()
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(6, 6, 6)
    expected_occluder = np.ones(shape=(6, 6, 6))
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


@pytest.mark.parametrize(
    "translation",
    [
        glm.vec3(0, 0, 0),
        glm.vec3(2, 2, 2),
        glm.vec3(4, 4, 4),
        glm.vec3(4, 0, 0),
        glm.vec3(0, 4, 0),
        glm.vec3(0, 0, 4),
        glm.vec3(4, 4, 0),
        glm.vec3(4, 0, 4),
        glm.vec3(0, 4, 4),
    ],
)
def test_global_occluder_blit_single_voxel(window_config: WindowConfig, translation: glm.vec3):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((1, 1, 1), window_config.ctx)
    vox_obj.translation = glm.vec3(1, 0, 1) + translation

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5), dtype=np.byte)
    pos = glm.ivec3(vox_obj.translation - glm.vec3(1, 0, 1))
    expected_occluder[pos[2]][pos[1]][pos[0]] = 1
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


@pytest.mark.parametrize(
    "translation",
    [
        glm.vec3(0, 0, 0),
        glm.vec3(2, 2, 2),
        glm.vec3(4, 4, 4),
        glm.vec3(4, 0, 0),
        glm.vec3(0, 4, 0),
        glm.vec3(0, 0, 4),
        glm.vec3(4, 4, 0),
        glm.vec3(4, 0, 4),
        glm.vec3(0, 4, 4),
    ],
)
def test_global_occluder_blit_occluder_translation(window_config: WindowConfig, translation: glm.vec3):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    global_occluder.occluder_volume.translation = -translation
    vox_obj = create_cube((1, 1, 1), window_config.ctx)
    vox_obj.translation = glm.vec3(1, 0, 1)

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5), dtype=np.byte)
    pos = glm.ivec3(translation)
    expected_occluder[pos[2]][pos[1]][pos[0]] = 1
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


def test_global_occluder_blit_single_voxel_on_edge(window_config: WindowConfig):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((1, 1, 1), window_config.ctx)
    vox_obj.translation = glm.vec3(0.501, 0, 0.501)

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5), dtype=np.byte)
    expected_occluder[0][0][0] = 1
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


def test_global_occluder_blit_single_voxel_on_edge_fails(window_config: WindowConfig):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((1, 1, 1), window_config.ctx)
    vox_obj.translation = glm.vec3(0.5, 0, 0.5)

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5), dtype=np.byte)
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


def test_global_occluder_blit_simple_2x2x2_edge(window_config: WindowConfig):
    """
    Top View of 2x2x2 cube centered (only x,z) at origin (translation = (0,0,0))

    +-------+
    |   |   |
    |---O---+--------------> x
    |   | 1 |
    +---+---+
        |
        |   Global Occluder
        |
        |
        v
        z

    Side View, object pivot is on min-y

        y
        ^
        |
        |   Global Occluder
        |
    +---+---+
    |   | 1 |
    |---+---+
    |   | 1 |
    +---O---+--------------> x
    """

    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((2, 2, 2), window_config.ctx)
    vox_obj.translation = glm.vec3(0, 0, 0)

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5))
    expected_occluder[0][0][0] = 1  # (0,0,0)
    expected_occluder[0][1][0] = 1  # (0,1,0)
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


@pytest.mark.parametrize(
    "translation",
    [
        glm.vec3(1, 1, 1),
        glm.vec3(3, 3, 3),
        glm.vec3(3, 1, 1),
        glm.vec3(1, 3, 1),
        glm.vec3(1, 1, 3),
        glm.vec3(3, 3, 1),
        glm.vec3(3, 1, 3),
        glm.vec3(1, 3, 3),
    ],
)
def test_global_occluder_blit_simple_2x2x2(window_config: WindowConfig, translation: glm.vec3):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((2, 2, 2), window_config.ctx)
    vox_obj.translation = glm.vec3(0, 0, 0) + translation

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5))
    pos = glm.ivec3(vox_obj.translation - glm.vec3(1, 0, 1))
    for ox in range(2):
        for oy in range(2):
            for oz in range(2):
                expected_occluder[pos[2] + ox][pos[1] + oy][pos[0] + oz] = 1
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


def test_global_occluder_blit_simple_2x2x2_then_clear(window_config: WindowConfig):
    # arrange
    translation = glm.vec3(1, 1, 1)
    global_occluder = GlobalOccluder(window_config, dimensions=(5, 5, 5))
    vox_obj = create_cube((2, 2, 2), window_config.ctx)
    vox_obj.translation = glm.vec3(0, 0, 0) + translation

    # act
    global_occluder.blit_object(vox_obj)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5))
    pos = glm.ivec3(vox_obj.translation - glm.vec3(1, 0, 1))
    for ox in range(2):
        for oy in range(2):
            for oz in range(2):
                expected_occluder[pos[2] + ox][pos[1] + oy][pos[0] + oz] = 1
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)

    # act
    global_occluder.clear()
    window_config.ctx.finish()
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(5, 5, 5)
    expected_occluder = np.zeros(shape=(5, 5, 5))
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


@pytest.mark.parametrize(
    ("min_cell", "max_cell"),
    [
        ((0, 0, 0), (1, 1, 1)),
        ((0, 0, 5), (1, 1, 6)),
        ((0, 5, 0), (1, 6, 1)),
        ((5, 0, 0), (6, 1, 1)),
        ((0, 5, 5), (1, 6, 6)),
        ((5, 0, 5), (6, 1, 6)),
        ((5, 5, 0), (6, 6, 1)),
        ((5, 5, 5), (6, 6, 6)),
        ((0, 0, 0), (6, 6, 6)),
        ((1, 2, 3), (4, 5, 6)),
    ],
)
def test_global_occluder_clear_region(
    window_config: WindowConfig,
    min_cell: tuple[int, int, int],
    max_cell: tuple[int, int, int],
):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(6, 6, 6))
    fill_occluder(global_occluder, window_config.ctx)

    # act
    global_occluder.clear_region(min_cell, max_cell)
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(6, 6, 6)
    expected_occluder = np.ones(shape=(6, 6, 6), dtype=np.byte)
    expected_occluder[min_cell[2] : max_cell[2], min_cell[1] : max_cell[1], min_cell[0] : max_cell[0]] = 0
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)


def test_global_occluder_clear_region_occluder_translation_has_no_influence(window_config: WindowConfig):
    # arrange
    global_occluder = GlobalOccluder(window_config, dimensions=(6, 6, 6))
    fill_occluder(global_occluder, window_config.ctx)
    global_occluder.occluder_volume.translation = glm.vec3(-3, -3, -3)

    # act
    global_occluder.clear_region((0, 0, 0), (1, 1, 1))
    window_config.ctx.finish()

    # assert
    resulting_occluder = np.frombuffer(global_occluder.occluder_texture.read(), dtype=np.byte).reshape(6, 6, 6)
    expected_occluder = np.ones(shape=(6, 6, 6), dtype=np.byte)
    expected_occluder[0][0][0] = 0
    np.testing.assert_array_equal(resulting_occluder, expected_occluder)
