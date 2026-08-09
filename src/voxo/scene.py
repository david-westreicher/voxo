from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
from moderngl import Context, Texture
from moderngl_window.context.base.window import WindowConfig
from moderngl_window.scene.camera import Camera
from pyglm import glm

from .objects import Light, Sun, VoxelObject, Water, World
from .utils import chunk_iters, frustum_cull_spheres, hdr_texture


@dataclass
class SkySetting:
    sun_direction: glm.vec3
    sun_color: glm.vec3
    sky_texture: Path


SUNNY = SkySetting(
    sun_direction=glm.normalize(glm.vec3(0.50, 0.72, 0.33)),
    sun_color=glm.vec3(4.4, 4.0, 3.0),
    sky_texture=Path("assets") / "kloofendal_48d_partly_cloudy_puresky_2k.hdr",
)
SUNDOWNER = SkySetting(
    sun_direction=glm.normalize(glm.vec3(0.60, 0.05, 0.33)),
    sun_color=glm.vec3(3.0, 1.5, 0.8),
    sky_texture=Path("assets") / "citrus_orchard_road_puresky_2k.hdr",
)
SKY_SETTING = SUNNY


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.voxel_objects: list[VoxelObject] = []
        self.lights: list[Light] = []
        self.waters: list[Water] = []
        self.last_frame_transforms: list[glm.mat4x4] = []
        self.ctx = ctx

        self.sun = Sun()
        self.sun.direction = SKY_SETTING.sun_direction
        self.sun.color = SKY_SETTING.sun_color

        self.world = World.from_file(Path("./resources/levels/test.lvl"))
        # self.world = World.from_file(Path("./resources/levels/carib_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/caveisland_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/ch_factory_fetch.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/ch_lee_fetch.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/ch_mall_fetch.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/cullington_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/frustrum_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/hub_carib_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/mansion_sandbox.lvl"))  # noqa: ERA001
        # self.world = World.from_file(Path("./resources/levels/marina_sandbox.lvl"))  # noqa: ERA001
        for light in self.world.lights:
            self.add_light(light)
        for water in self.world.waters:
            self.add_water(water)
        self.world.texture_information.upload_to_gpu(ctx)
        self.object_generator = chunk_iters(self.world.voxel_objects, 4000)

    def add_voxel_object(self, voxel_object: VoxelObject) -> VoxelObject:
        self.voxel_objects.append(voxel_object)
        voxel_object.upload_to_gpu(self.ctx)
        return voxel_object

    def add_light(self, light: Light) -> Light:
        self.lights.append(light)
        light.upload_to_gpu()
        return light

    def add_water(self, water: Water) -> Water:
        self.waters.append(water)
        water.upload_to_gpu()
        return water

    @cached_property
    def suns(self) -> Sequence[Sun]:
        return [self.sun]

    def update(self, time: float) -> None:  # noqa: ARG002
        objs = next(self.object_generator, None)
        if objs is not None:
            for obj in objs:
                self.add_voxel_object(obj)

    def visible_objects(self, camera: Camera) -> list[VoxelObject]:
        bounding_spheres = [obj.bounding_sphere for obj in self.voxel_objects]
        return [
            obj
            for obj, vis in zip(
                self.voxel_objects,
                frustum_cull_spheres(camera.matrix, camera.projection.matrix, bounding_spheres),
                strict=True,
            )
            if vis and obj.visible
        ]


@cache
def global_skybox(window: WindowConfig) -> Texture:
    def remove_sun(image: np.ndarray) -> np.ndarray:
        return np.array(np.minimum(image, 200.0))

    assert window.resource_dir
    sky_texture = hdr_texture(
        window.resource_dir / SKY_SETTING.sky_texture,
        window.ctx,
        post_processing=remove_sun,
    )
    sky_texture.label = "texture2d_hdr_sky"
    return sky_texture
