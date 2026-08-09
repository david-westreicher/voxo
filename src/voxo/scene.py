from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from moderngl import Context
from moderngl_window.scene.camera import Camera
from pyglm import glm

from .objects import Light, Sun, VoxelObject, Water, World
from .utils import chunk_iters, frustum_cull_spheres


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.voxel_objects: list[VoxelObject] = []
        self.lights: list[Light] = []
        self.waters: list[Water] = []
        self.last_frame_transforms: list[glm.mat4x4] = []
        self.ctx = ctx

        self.sun = Sun()
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

    def update(self, time: float) -> None:
        objs = next(self.object_generator, None)
        if objs is not None:
            for obj in objs:
                self.add_voxel_object(obj)

        self.sun.direction = glm.normalize(glm.vec3(glm.sin(time), 1, glm.cos(time)))

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
