from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from moderngl import Context
from moderngl_window.scene.camera import Camera
from pyglm import glm

from .model import VoxLight, parse_xml_level
from .objects import AreaLight, ConeLight, Light, SphereLight, Sun, VoxelObject, World
from .utils import chunk_iters, frustum_cull_spheres


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.voxel_objects: list[VoxelObject] = []
        self.lights: list[Light] = []
        self.last_frame_transforms: list[glm.mat4x4] = []
        self.ctx = ctx

        self.sun = Sun()
        # self.world = World.from_file(Path("./resources/levels/carib_sandbox.lvl"))
        self.world = World.from_file(Path("./test.lvl"))
        vox_lights = [
            obj for obj in parse_xml_level(Path("./resources/levels/carib_sandbox/test.xml")) if type(obj) is VoxLight
        ]
        for vox_light in vox_lights:
            assert vox_light.reach > 0.0
            light: Light = SphereLight()
            if vox_light.light_type == "area":
                light = AreaLight(size=vox_light.size)
            elif vox_light.light_type == "cone":
                light = ConeLight(penumbra=vox_light.penumbra, reach=vox_light.reach)
            else:
                light = SphereLight()
            light.reach = vox_light.reach
            light.intensity = vox_light.scale
            light.translation = vox_light.translation
            light.rotation = vox_light.rotation
            light.color = vox_light.color
            self.add_light(light)
        self.world.texture_information.upload_to_gpu(ctx)
        self.object_generator = chunk_iters(self.world.voxel_objects, 4000)

    def add_voxel_object(self, voxel_object: VoxelObject) -> VoxelObject:
        self.voxel_objects.append(voxel_object)
        voxel_object.upload_to_gpu(self.ctx)
        return voxel_object

    def add_light(self, light: Light) -> Light:
        self.lights.append(light)
        return light

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
