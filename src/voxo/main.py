from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import moderngl_window
from moderngl_window.context.base import KeyModifiers
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm

from .constants import ASPECT_RATIO, CENTER, GLOBAL_OCCLUDER_DIMENSIONS, SCREEN_DIMENSIONS
from .rendering import GBufferDebug, GBufferPingPong, WireFrameRenderer
from .scene import Scene
from .voxel_rendering import GlobalOccluder, VoxelLighting, VoxelRenderer

if TYPE_CHECKING:
    from pyglm.glm import mat4x4 as Mat4  # noqa: N812


class CameraWindow(moderngl_window.WindowConfig):  # type: ignore[misc, name-defined]
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=ASPECT_RATIO, near=1.0, far=1000.0, fov=74)
        self.camera.mouse_sensitivity = 0.05
        self.camera.velocity = 50.0
        self.camera_enabled = True

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.LEFT_SHIFT:
                self.camera.velocity = 1
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

        if action == keys.ACTION_RELEASE and key == keys.LEFT_SHIFT:
            self.camera.velocity = 50.0

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:  # noqa: ARG002
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def on_resize(self, width: int, height: int) -> None:  # noqa: ARG002
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:  # noqa: ARG002
        velocity = self.camera.velocity + y_offset
        self.camera.velocity = max(velocity, 1.0)


class VoxoWindow(CameraWindow):
    gl_version = (4, 6)
    window_size = SCREEN_DIMENSIONS
    title = "voxo"
    resource_dir = Path("resources").resolve()
    vsync = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.time = 0.0
        self.frame_counter = 0
        self.debug = False
        self.camera.position = glm.vec3(CENTER)  # type:ignore[assignment]
        self.scene = Scene(self.ctx)

        self.last_frame_projview: Mat4 = cast("Mat4", self.camera.projection.matrix @ self.camera.matrix)
        self.global_occluder = GlobalOccluder(self, GLOBAL_OCCLUDER_DIMENSIONS)
        self.voxel_renderer = VoxelRenderer(self)
        self.gbuffer = GBufferPingPong(self, SCREEN_DIMENSIONS)
        self.gbuffer_lighting = VoxelLighting(self, SCREEN_DIMENSIONS)
        self.gbuffer_debug = GBufferDebug(self, self.gbuffer_lighting.lighting_texture)
        self.wireframe_box = WireFrameRenderer(self)

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_RELEASE and key == keys.B:
            self.debug = not self.debug

    def on_render(self, time: float, frametime: float) -> None:  # noqa: ARG002
        self.time = time
        self.frame_counter += 1
        if self.timer.is_running:
            self.scene.update(time)

            # Update Occluder
            self.global_occluder.clear()
            for voxel_object in self.scene.voxel_objects:
                self.global_occluder.blit_object(voxel_object)
            self.global_occluder.update_mipmaps()

        # Render into HDR framebuffer
        gbuffer = self.gbuffer.current
        gbuffer.start()
        for i, voxel_object in enumerate(self.scene.voxel_objects):
            self.voxel_renderer.render(
                self.camera,
                voxel_object,
                self.scene.last_frame_transforms[i],
                self.last_frame_projview,
                self.frame_counter,
            )
        self.last_frame_projview = cast("Mat4", self.camera.projection.matrix @ self.camera.matrix)
        self.scene.update_lastframe_transforms()

        # Compute lighting
        gbuffer.smooth_normals(self.camera)
        self.gbuffer_lighting.render(
            self.camera,
            gbuffer,
            self.global_occluder.occluder_texture,
            self.frame_counter,
            self.scene.light.translation,
            self.gbuffer.last.linear_depth,
        )

        # Render framebuffer onto screen
        self.ctx.screen.use()
        self.gbuffer_debug.render(gbuffer, debug=self.debug)
        if self.debug:
            self.global_occluder.render_debug(self.camera)
            self.wireframe_box.render(self.camera, self.scene.voxel_objects)
            self.wireframe_box.render(self.camera, [self.scene.light, self.global_occluder.occluder_volume])
        self.gbuffer.swap()
