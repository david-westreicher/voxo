from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import glm
import moderngl
import moderngl_window
from glm import mat4 as Mat4  # noqa: N812
from moderngl_window import geometry
from moderngl_window.context.base import KeyModifiers
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene.camera import KeyboardCamera


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
        return self.translation @ self.rotation @ self.scale


class CubeSimple(CameraWindow):
    gl_version = (4, 6)
    title = "Plain Cube"
    resource_dir = Path("resources").resolve()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.prog = self.load_program("programs/cube_simple.glsl")
        self.prog["color"].value = 1.0, 1.0, 0.0, 1.0

        self.cube = Object(geometry.cube(size=(2, 2, 2)))
        self.cube.translation = glm.translate(glm.vec3(0.0, -1.0, 0.0))
        self.cube.scale = glm.scale(glm.vec3(10.0, 0.1, 10.0))

    def on_render(self, time: float, frametime: float) -> None:  # noqa: ARG002
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        self.prog["m_proj"].write(self.camera.projection.matrix)
        self.prog["m_model"].write(self.cube.modelview)
        self.prog["m_camera"].write(self.camera.matrix)

        self.cube.geometry.render(self.prog)
