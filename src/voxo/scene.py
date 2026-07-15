from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from moderngl import Context
from pyglm import glm

from .constants import CENTER, CENTER_GROUND
from .model import parse_model
from .objects import Light, Sun, VoxelObject


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.truck = VoxelObject(model=parse_model(Path("./resources/models/duck.txt")))
        self.dwarf = VoxelObject(model=parse_model(Path("./resources/models/dwarf.txt")))
        self.plane = VoxelObject(model=parse_model(Path("./resources/models/corner.txt")))
        self.light_1 = Light(10.0, glm.vec3(20.0, 18.0, 15.0) * 800.0)
        self.light_2 = Light(0.1, glm.vec3(20.0, 1.0, 1.0) * 500.0)
        self.light_3 = Light(5.0, glm.vec3(1.0, 1.0, 20.0) * 500.0)
        self.light_4 = Light(1.0, glm.vec3(0.0, 0.0, 0.0))
        self.sun = Sun()

        self.last_frame_transforms = [obj.transform for obj in self.voxel_objects]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(ctx)

    @cached_property
    def voxel_objects(self) -> Sequence[VoxelObject]:
        return [self.dwarf, self.plane, self.truck]

    @cached_property
    def lights(self) -> Sequence[Light]:
        return [self.light_2]  # , self.light_2, self.light_3]

    @cached_property
    def suns(self) -> Sequence[Sun]:
        return [self.sun]

    def update(self, time: float) -> None:
        # TODO(david): occluder should align to +/-0.5 voxel
        self.plane.translation = glm.vec3(0, 0, 0)
        self.truck.translation = glm.vec3(0, 1, time * 0.1) + glm.vec3(10, 0, 30)
        self.dwarf.translation = glm.vec3(0, 1, 0) + glm.vec3(30, 0, 30)
        self.light_1.translation = glm.rotateY(glm.vec3(80, 0.0, 0), time) + CENTER
        self.light_1.radius = glm.sin(time) * 20.0
        self.light_2.translation = glm.rotateY(glm.vec3(0, 0, 30), time * 2.0) + glm.vec3(30, 50, 30)
        self.light_3.translation = glm.rotateY(glm.vec3(80, -20.0, 0), (time + glm.pi() * 0.5) * 3.0) + CENTER
        self.sun.direction = glm.normalize(glm.vec3(glm.sin(time), 1, glm.cos(time)))
        # self.truck.rotate(0.001, glm.vec3(0, 1, 0))
        # self.dwarf.rotate(-0.001, glm.vec3(0, 1, 0))

    def update_lastframe_transforms(self) -> None:
        for i, voxel_object in enumerate(self.voxel_objects):
            self.last_frame_transforms[i] = voxel_object.transform
