from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from moderngl import Context
from pyglm import glm

from .model import parse_model
from .objects import Light, Sun, VoxelObject


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.corner_left = VoxelObject(model=parse_model(Path("./resources/models/corner.txt")), name="corner_left")
        self.corner_right = VoxelObject(model=parse_model(Path("./resources/models/corner.txt")), name="corner_right")
        self.corner_front = VoxelObject(model=parse_model(Path("./resources/models/corner.txt")), name="corner_front")
        self.plane_1 = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.plane_2 = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.plane_3 = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.plane_4 = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.truck_1 = VoxelObject(model=parse_model(Path("./resources/models/truck.txt")))
        self.truck_2 = VoxelObject(model=parse_model(Path("./resources/models/truck.txt")))
        # ./../../resources/models/treehouse.txt
        self.dwarf = VoxelObject(model=parse_model(Path("./resources/models/treehouse.txt")))
        self.light_1 = Light(1.0, glm.vec3(20.0, 18.0, 15.0) * 800.0)
        self.light_2 = Light(5.0, glm.vec3(20.0, 1.0, 1.0) * 500.0)
        self.light_3 = Light(5.0, glm.vec3(1.0, 1.0, 20.0) * 500.0)
        self.sun = Sun()

        self.corner_left.translation = glm.vec3(64, 0, 64)
        self.corner_right.translation = glm.vec3(256 + 64, 0, 64)
        self.corner_right.rotation = glm.angleAxis(glm.pi() * 1.5, glm.vec3(0, 1, 0))
        self.corner_front.rotation = glm.angleAxis(glm.pi(), glm.vec3(0, 1, 0))
        self.corner_front.translation = glm.vec3(256 + 64, 0, 128 + 64)
        self.plane_1.translation = glm.vec3(64 + 1, 0, 64 + 127)
        self.plane_2.translation = glm.vec3(64 + 128 - 1, 0, 64 + 127)
        self.plane_3.translation = glm.vec3(64 + 1, 70, 64)
        self.plane_4.translation = glm.vec3(64 + 128, 70, 64)

        self.truck_1.translation = glm.vec3(95, 1, 60)
        self.truck_2.translation = glm.vec3(128, 1, 200)
        self.dwarf.translation = glm.vec3(128, 1, 190)

        self.voxel_objects = [
            self.corner_left,
            self.corner_right,
            self.corner_front,
            self.plane_1,
            self.plane_2,
            self.plane_3,
            self.plane_4,
            self.truck_1,
            # self.truck_2,
            self.dwarf,
        ]

        self.last_frame_transforms = [obj.transform for obj in self.voxel_objects]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(ctx)

        self.ctx = ctx

    def add_voxel_object(self, voxel_object: VoxelObject) -> VoxelObject:
        self.voxel_objects.append(voxel_object)
        voxel_object.upload_to_gpu(self.ctx)
        self.last_frame_transforms.append(voxel_object.transform)
        return voxel_object

    @cached_property
    def lights(self) -> Sequence[Light]:
        return [self.light_1]

    @cached_property
    def suns(self) -> Sequence[Sun]:
        return [self.sun]

    def update(self, time: float) -> None:
        # TODO(david): occluder should align to +/-0.5 voxel

        self.sun.direction = glm.normalize(glm.vec3(glm.sin(time), 1, glm.cos(time)))
        self.light_1.translation = glm.vec3(140, 56, 180) + glm.rotateY(glm.vec3(10, 0, 0), time)

    def update_lastframe_transforms(self) -> None:
        for i, voxel_object in enumerate(self.voxel_objects):
            self.last_frame_transforms[i] = voxel_object.transform
