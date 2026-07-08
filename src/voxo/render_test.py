from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import moderngl
import moderngl_window
import numpy as np
from moderngl import Program
from moderngl_window import geometry
from moderngl_window.context.base import KeyModifiers
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene import Camera
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812

SIZE = 100


class CameraWindow(moderngl_window.WindowConfig):  # type: ignore[misc, name-defined]
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera.mouse_sensitivity = 0.05
        self.camera_enabled = True

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:  # noqa: ARG002
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def on_resize(self, width: int, height: int) -> None:  # noqa: ARG002
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:  # noqa: ARG002
        velocity = self.camera.velocity + y_offset
        self.camera.velocity = max(velocity, 1.0)


@dataclass
class Object:
    geometry: VAO
    rotation: Mat4 = field(default=glm.mat4(1.0))
    translation: Mat4 = field(default=glm.translate(glm.vec3(0.0)))
    scale: Mat4 = field(default=glm.scale(glm.vec3(1.0)))

    @property
    def modelview(self) -> Mat4:
        return cast("Mat4", self.translation @ self.rotation @ self.scale)


class VoxelRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.program: Program = window.load_program("programs/raytrace_voxels.glsl")
        self.geometry: VAO = geometry.quad_fs(normals=False, uvs=True)
        data = np.random.randint(0, 256, size=(SIZE, SIZE, SIZE), dtype=np.uint8).tobytes()  # noqa: NPY002
        #  data = np.zeros(shape=(SIZE, SIZE, SIZE), dtype=np.uint8)
        # np.fill_diagonal(data, 255)
        self.voxel_texture = window.ctx.texture3d((SIZE, SIZE, SIZE), data=data, components=1, alignment=1, dtype="f1")
        self.voxel_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.voxel_texture.repeat_x = False
        self.voxel_texture.repeat_y = False
        self.voxel_texture.repeat_z = False

    def render(self, camera: Camera) -> None:
        ctx = self.voxel_texture.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))  # type:ignore[union-attr]
        self.program["uInvView"].write(glm.inverse(camera.matrix))  # type:ignore[union-attr]
        self.program["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.program["u_voxel_data"].value = 0  # type:ignore[union-attr]

        self.voxel_texture.use(location=0)
        self.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


class CubeSimple(CameraWindow):
    gl_version = (4, 6)
    title = "Plain Cube"
    resource_dir = Path("resources").resolve()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.full_screen_shader = VoxelRenderer(self)
        self.prog = self.load_program("programs/cube_simple.glsl")
        self.prog["color"].value = 1.0, 1.0, 0.0, 1.0

        self.cube = Object(geometry.cube(size=(1, 1, 1)))
        self.cube.translation = glm.translate(glm.vec3(0.0))
        self.cube.scale = glm.scale(glm.vec3(SIZE))

    def on_render(self, time: float, frametime: float) -> None:  # noqa: ARG002
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        self.prog["m_proj"].write(self.camera.projection.matrix)
        self.prog["m_model"].write(self.cube.modelview)
        self.prog["m_camera"].write(self.camera.matrix)

        self.full_screen_shader.render(self.camera)

        self.ctx.wireframe = True
        self.cube.geometry.render(self.prog)
        self.ctx.wireframe = False
