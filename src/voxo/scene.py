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
        self.plane_1 = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.dwarf = VoxelObject(model=parse_model(Path("./resources/models/dwarf.txt")))
        self.light_1 = Light(10.0, glm.vec3(20.0, 18.0, 15.0) * 800.0)
        self.light_2 = Light(5.0, glm.vec3(20.0, 1.0, 1.0) * 500.0)
        self.light_3 = Light(5.0, glm.vec3(1.0, 1.0, 20.0) * 500.0)
        self.sun = Sun()

        self.last_frame_transforms = [obj.transform for obj in self.voxel_objects]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(ctx)

    @cached_property
    def voxel_objects(self) -> Sequence[VoxelObject]:
        return [
            self.plane_1,
            self.dwarf,
        ]

    @cached_property
    def lights(self) -> Sequence[Light]:
        return []  # self.light_2]

    @cached_property
    def suns(self) -> Sequence[Sun]:
        return [self.sun]

    def update(self, time: float) -> None:
        self.plane_1.translation = glm.vec3(64, 0, 64)

        self.dwarf.translation = glm.vec3(64, 1, 64)

        self.sun.direction = glm.normalize(glm.vec3(glm.sin(time), 1, glm.cos(time)))
        self.light_1.translation = glm.rotateY(glm.vec3(80, 0.0, 0), time) + CENTER
        self.light_1.radius = glm.sin(time) * 20.0
        self.light_2.translation = glm.rotateY(glm.vec3(80, -20.0, 0), time * 3.0) + CENTER
        self.light_3.translation = glm.rotateY(glm.vec3(80, -20.0, 0), (time + glm.pi() * 0.5) * 3.0) + CENTER

    def update_lastframe_transforms(self) -> None:
        for i, voxel_object in enumerate(self.voxel_objects):
            self.last_frame_transforms[i] = voxel_object.transform
