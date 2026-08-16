from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import moderngl_window
from moderngl_window.context.base import KeyModifiers
from moderngl_window.scene import Camera
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm

from .constants import (
    ASPECT_RATIO,
    CAMERA_FAR,
    CAMERA_FOV,
    CAMERA_NEAR,
    CENTER,
    GLOBAL_OCCLUDER_DIMENSIONS,
    SCREEN_DIMENSIONS,
)
from .debug import DebugView
from .objects import VoxelObjectGPUBuffer
from .rendering import GBufferPingPong, PostProcessing, WaterRenderer, WireFrameRenderer
from .scene import Scene, global_skybox
from .utils import Timer
from .voxel_rendering import GlobalOccluder, VoxelLighting, VoxelRenderer

if TYPE_CHECKING:
    from pyglm.glm import mat4x4 as Mat4  # noqa: N812


class CameraWindow(moderngl_window.WindowConfig):  # type: ignore[misc, name-defined]
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(
            self.wnd.keys,
            aspect_ratio=ASPECT_RATIO,
            near=CAMERA_NEAR,
            far=CAMERA_FAR,
            fov=CAMERA_FOV,
        )
        self.camera.mouse_sensitivity = 0.05
        self.camera.velocity = 500.0
        self.camera_enabled = True

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.LEFT_SHIFT:
                self.camera.velocity = 20
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

        if action == keys.ACTION_RELEASE and key == keys.LEFT_SHIFT:
            self.camera.velocity = 500.0

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
    aspect_ratio = None
    window_size = SCREEN_DIMENSIONS
    title = "voxo"
    resource_dir = Path("resources").resolve()
    vsync = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.time = 0.0
        self.frame_counter = 0
        self.global_timer = Timer.global_timer()
        self.debug = False
        self.synced_camera = Camera(fov=CAMERA_FOV, aspect_ratio=ASPECT_RATIO, near=CAMERA_NEAR, far=CAMERA_FAR)
        self.camera.position = glm.vec3(CENTER) + glm.vec3(0, 100, 0)

        self.last_frame_projview: Mat4 = cast("Mat4", self.camera.projection.matrix @ self.camera.matrix)
        self.voxel_object_gpu_buffer = VoxelObjectGPUBuffer(self.ctx)
        self.global_occluder = GlobalOccluder(self, GLOBAL_OCCLUDER_DIMENSIONS, center=True)
        self.voxel_renderer = VoxelRenderer(self)
        self.gbuffer = GBufferPingPong(self, SCREEN_DIMENSIONS)
        self.voxel_lighting = VoxelLighting(self, SCREEN_DIMENSIONS, global_skybox(self))
        self.wireframe_box = WireFrameRenderer(self)
        self.water_renderer = WaterRenderer(self, SCREEN_DIMENSIONS)
        self.post_processing = PostProcessing(self, SCREEN_DIMENSIONS, global_skybox(self))

        self.scene = Scene(self.ctx, self.argv.start_level)
        self.debugger = DebugView(
            self,
            self.scene,
            self.camera,
            [
                *self.gbuffer.textures,
                *self.voxel_lighting.textures,
                *self.post_processing.textures,
                *self.water_renderer.textures,
                self.scene.world.texture_information.material_texture,
                self.scene.world.texture_information.palette_texture,
            ],
            [
                *self.gbuffer.shaders,
                *self.voxel_lighting.shaders,
                *self.post_processing.shaders,
                *self.voxel_renderer.shaders,
                *self.water_renderer.shaders,
            ],
        )

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--start_level", type=Path)

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
        with (
            self.ctx.debug_scope(name),
            self.debugger.gpu_profiler.query(name, self.frame_counter),
            self.debugger.cpu_profiler.query(name),
        ):
            yield

    def sync_camera(self, camera: Camera) -> None:
        _ = camera.matrix
        _ = camera.projection.matrix
        self.synced_camera.position.x = camera.position.x
        self.synced_camera.position.y = camera.position.y
        self.synced_camera.position.z = camera.position.z
        self.synced_camera.yaw = camera.yaw
        self.synced_camera.pitch = camera.pitch

    def on_render(self, time: float, frametime: float) -> None:
        self.sync_camera(self.camera)
        self.time = time
        self.frame_counter += 0 if self.debugger.is_frame_counter_stopped else 1

        if self.timer.is_running or not self.camera_enabled:
            self.scene.update(time)

        # Update Occluder
        occluder_was_updated = False
        with self.profile("update occluder"):
            for voxel_object in self.scene.voxel_objects:
                if voxel_object.visible and voxel_object.last_frame_update >= self.global_timer.time:
                    # TODO(david): clear dirty objects
                    self.global_occluder.blit_object(voxel_object)
                    occluder_was_updated = True
        with self.profile("occluder mipmaps"):
            if occluder_was_updated:
                self.global_occluder.update_mipmaps()

        # Fill GBuffer
        with self.debugger.cpu_profiler.query("visible objects"):
            # TODO(david): Replace this with GPU culling
            visible_objects = [obj for obj in self.scene.voxel_objects if obj.visible]
            self.voxel_object_gpu_buffer.update_gpu_buffers(self.scene.voxel_objects, self.global_timer.time)
            visible_objects.sort(key=lambda obj: glm.distance2(self.synced_camera.position, obj.translation))

        with self.profile("fill gbuffer"):
            gbuffer = self.gbuffer.current
            gbuffer.start()
            self.voxel_renderer.render_objects(
                self.voxel_object_gpu_buffer,
                visible_objects,
                self.synced_camera,
                self.last_frame_projview,
                self.gbuffer.current.linear_depth,
                self.frame_counter,
            )
            for obj in self.scene.voxel_objects:
                obj.last_frame_transform = obj.transform

        with self.profile("smooth normals"):
            # TODO(david): Currently unused because water renders into normal buffer
            gbuffer.smooth_normals(self.synced_camera)

        with self.profile("water"):
            self.water_renderer.copy_albedo(self.gbuffer.current)
            gbuffer.framebuffer.use()
            self.water_renderer.render(
                self.synced_camera,
                self.gbuffer.current,
                self.scene.waters,
                self.frame_counter,
            )

        # Compute lighting
        with self.profile("ambient"):
            self.voxel_lighting.clear()
            self.voxel_lighting.render_ambient(
                self.synced_camera,
                self.gbuffer.current,
                self.global_occluder,
                self.frame_counter,
            )
        with self.profile("direct"):
            self.voxel_lighting.render_direct(
                self.synced_camera,
                self.gbuffer.current,
                self.global_occluder,
                self.scene.lights,
                self.scene.suns,
                self.frame_counter,
            )
        with self.profile("denoise direct"):
            self.voxel_lighting.denoise_direct(
                self.synced_camera,
                self.gbuffer.current,
                self.gbuffer.last,
                self.frame_counter,
                camera_moved=self.last_frame_projview
                != (self.synced_camera.projection.matrix @ self.synced_camera.matrix),
            )
        with self.profile("specular"):
            self.voxel_lighting.render_specular(
                self.synced_camera,
                self.gbuffer.current,
                self.gbuffer.last,
                self.global_occluder,
                self.scene.suns,
                self.frame_counter,
                camera_moved=self.last_frame_projview
                != (self.synced_camera.projection.matrix @ self.synced_camera.matrix),
            )

        # Post processing
        with self.profile("post processing"):
            self.post_processing.render(
                camera=self.synced_camera,
                motion_vectors=self.gbuffer.current.motion_vectors,
                suns=self.scene.suns,
                light_texture=self.voxel_lighting.final_light_texture,
                depth_texture=self.gbuffer.current.depth_texture,
                linear_depth_texture=self.gbuffer.current.linear_depth,
                prev_linear_depth_texture=self.gbuffer.last.linear_depth,
            )
            self.ctx.screen.use()
            self.post_processing.render_final_tonemapped_texture()

        # Render Debug Information
        if self.debug:
            self.global_occluder.render_debug(self.synced_camera)
            self.wireframe_box.render(self.synced_camera, self.scene.voxel_objects)
            self.wireframe_box.render(
                self.synced_camera,
                [
                    self.global_occluder.occluder_volume,
                    *self.scene.lights,
                    *self.scene.waters,
                ],
            )
        if not self.camera_enabled:
            self.debugger.render_debug(frametime)
        self.gbuffer.swap()
        self.last_frame_projview = cast("Mat4", self.synced_camera.projection.matrix @ self.synced_camera.matrix)
        self.global_timer.tick()
