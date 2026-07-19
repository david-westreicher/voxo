from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import moderngl_window
from moderngl_window.context.base import KeyModifiers
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm

from .constants import ASPECT_RATIO, CENTER, GLOBAL_OCCLUDER_DIMENSIONS, SCREEN_DIMENSIONS
from .debug import DebugView
from .rendering import GBufferDebug, GBufferPingPong, PostProcessing, WireFrameRenderer
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
    vsync = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.time = 0.0
        self.frame_counter = 0
        self.global_frame_counter = 0
        self.debug = False
        self.camera.position = glm.vec3(CENTER)
        self.scene = Scene(self.ctx)

        self.last_frame_projview: Mat4 = cast("Mat4", self.camera.projection.matrix @ self.camera.matrix)
        self.global_occluder = GlobalOccluder(self, GLOBAL_OCCLUDER_DIMENSIONS)
        self.voxel_renderer = VoxelRenderer(self)
        self.gbuffer = GBufferPingPong(self, SCREEN_DIMENSIONS)
        self.voxel_lighting = VoxelLighting(self, SCREEN_DIMENSIONS)
        self.gbuffer_debug = GBufferDebug(self)
        self.wireframe_box = WireFrameRenderer(self)
        self.post_processing = PostProcessing(self, SCREEN_DIMENSIONS)

        self.debugger = DebugView(
            self,
            self.scene,
            [
                *self.gbuffer.textures,
                *self.voxel_lighting.textures,
                *self.post_processing.textures,
            ],
            [
                *self.gbuffer.shaders,
                *self.voxel_lighting.shaders,
                *self.post_processing.shaders,
                *self.voxel_renderer.shaders,
            ],
        )

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.debugger.resize(width, height)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        super().on_mouse_position_event(x, y, dx, dy)
        self.debugger.mouse_position_event(x, y, dx, dy)  # type:ignore[no-untyped-call]

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.debugger.mouse_drag_event(x, y, dx, dy)  # type:ignore[no-untyped-call]

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        super().on_mouse_scroll_event(x_offset, y_offset)
        self.debugger.mouse_scroll_event(x_offset, y_offset)  # type:ignore[no-untyped-call]

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.debugger.mouse_press_event(x, y, button)  # type:ignore[no-untyped-call]

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        self.debugger.mouse_release_event(x, y, button)

    def on_unicode_char_entered(self, char: str) -> None:
        self.debugger.unicode_char_entered(char)  # type:ignore[no-untyped-call]

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        self.debugger.key_event(key, action, modifiers)  # type:ignore[no-untyped-call]
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_RELEASE and key == keys.B:
            self.debug = not self.debug

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        with self.ctx.debug_scope(name), self.debugger.profiler.query(name, self.frame_counter):
            yield

    def on_render(self, time: float, frametime: float) -> None:
        self.time = time
        self.global_frame_counter += 1
        self.frame_counter += 0 if self.debugger.is_frame_counter_stopped else 1
        with self.profile("update occluder"):
            if self.timer.is_running or not self.camera_enabled:
                self.scene.update(time)

                # Update Occluder
                self.global_occluder.clear()
                for voxel_object in self.scene.voxel_objects:
                    if voxel_object.visible:
                        self.global_occluder.blit_object(voxel_object)
                self.global_occluder.update_mipmaps()

        # Fill GBuffer
        with self.profile("fill gbuffer"):
            gbuffer = self.gbuffer.current
            gbuffer.start()
            self.voxel_renderer.render_objects(
                self.camera,
                self.scene.voxel_objects,
                self.scene.last_frame_transforms,
                self.last_frame_projview,
                self.frame_counter,
            )
            self.scene.update_lastframe_transforms()

        with self.profile("smooth normals"):
            gbuffer.smooth_normals(self.camera)

        # Compute lighting
        with self.profile("compute lighting"):
            self.voxel_lighting.render(
                self.camera,
                gbuffer,
                self.global_occluder.occluder_texture,
                self.scene.lights,
                self.scene.suns,
                self.frame_counter,
            )

        # Post processing
        with self.profile("post processing"):
            self.post_processing.render(
                camera=self.camera,
                camera_moved=self.last_frame_projview != (self.camera.projection.matrix @ self.camera.matrix),
                suns=self.scene.suns,
                irradiance=self.voxel_lighting.irradiance_texture,
                specular=self.voxel_lighting.specular_texture,
                current_gbuffer=self.gbuffer.current,
                last_gbuffer=self.gbuffer.last,
                frame_counter=self.frame_counter,
            )

        # Render framebuffer onto screen
        self.ctx.screen.use()
        self.gbuffer_debug.render(gbuffer, final_hdr_texture=self.post_processing.final_texture, debug=self.debug)
        self.wireframe_box.render(self.camera, [*self.scene.lights])
        if self.debug:
            self.global_occluder.render_debug(self.camera)
            self.wireframe_box.render(self.camera, self.scene.voxel_objects)
            self.wireframe_box.render(self.camera, [*self.scene.lights, self.global_occluder.occluder_volume])
        if not self.camera_enabled:
            self.debugger.render_debug(self.global_frame_counter, frametime)
        self.gbuffer.swap()
        self.last_frame_projview = cast("Mat4", self.camera.projection.matrix @ self.camera.matrix)
