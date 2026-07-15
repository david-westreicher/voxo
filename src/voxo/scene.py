from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from moderngl import Context
from pyglm import glm

from .constants import CENTER, CENTER_GROUND
from .model import parse_model
from .utils import Light, Sun
from .voxel_rendering import VoxelObject


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.truck = VoxelObject(model=parse_model(Path("./resources/models/truck.txt")))
        self.dwarf = VoxelObject(model=parse_model(Path("./resources/models/dwarf.txt")))
        self.plane = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.light_1 = Light(10.0, glm.vec3(20.0, 18.0, 15.0) * 800.0)
        self.light_2 = Light(5.0, glm.vec3(20.0, 1.0, 1.0) * 500.0)
        self.light_3 = Light(5.0, glm.vec3(1.0, 1.0, 20.0) * 500.0)
        self.sun = Sun()

        self.last_frame_transforms = [obj.transform for obj in self.voxel_objects]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(ctx)

    @cached_property
    def voxel_objects(self) -> Sequence[VoxelObject]:
        return [self.truck, self.dwarf, self.plane]

    @cached_property
    def lights(self) -> Sequence[Light]:
        return [self.light_1, self.light_2, self.light_3]

    @cached_property
    def suns(self) -> Sequence[Sun]:
        return [self.sun]

    def update(self, time: float) -> None:
        # TODO(david): occluder should align to +/-0.5 voxel
        self.truck.translation = glm.vec3(0, self.truck.model.dimensions[2] * 0.5 + 1, 0) + CENTER_GROUND
        self.dwarf.translation = glm.vec3(0, 58.5, 0) + CENTER_GROUND
        self.light_1.translation = glm.rotateY(glm.vec3(80, 0.0, 0), time) + CENTER
        self.light_1.radius = glm.sin(time) * 20.0
        self.light_2.translation = glm.rotateY(glm.vec3(80, -20.0, 0), time * 3.0) + CENTER
        self.light_3.translation = glm.rotateY(glm.vec3(80, -20.0, 0), (time + glm.pi() * 0.5) * 3.0) + CENTER
        self.plane.translation = CENTER_GROUND + glm.vec3(0, 0.5, 0)
        self.truck.rotate(0.001, glm.vec3(0, 1, 0))
        self.dwarf.rotate(-0.001, glm.vec3(0, 1, 0))

    def update_lastframe_transforms(self) -> None:
        for i, voxel_object in enumerate(self.voxel_objects):
            self.last_frame_transforms[i] = voxel_object.transform
