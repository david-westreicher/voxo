from pathlib import Path
from typing import Any, cast

import moderngl
import moderngl_window
from moderngl import Framebuffer, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.context.base import KeyModifiers
from moderngl_window.scene import Camera
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812
from pyglm.glm import vec3 as Vec3  # noqa: N812

from .constants import ASPECT_RATIO, CENTER, GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS, SCREEN_DIMENSIONS
from .rendering import GBuffer, GBufferDebug, GBufferPingPong, WireFrameRenderer
from .scene import Scene
from .voxel import GlobalOccluder, VoxelRenderer


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


class GBufferLighting:
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        size: tuple[int, int],
    ) -> None:
        self.pingpong = 0
        self.framebuffers: list[Framebuffer] = []
        for i in range(2):
            self.framebuffers.append(
                window.ctx.framebuffer(color_attachments=[window.ctx.texture(size=size, components=3, dtype="f2")])
            )
            self.framebuffers[-1].label = f"framebuffer_gbuffer_lighting_{i}"
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_lighting = window.load_program("programs/gbuffer_lighting.glsl", defines=GLOBAL_DEFINE)
        self.gbuffer_lighting.label = "prog_gbuffer_lighting"

        self.stbnormals = window.load_texture_array("assets/stbn_cosine_normals.png", layers=64)
        self.stbnormals.label = "texarr_stbn_cosine_normals"
        self.stbnormals.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.random_vec2 = window.load_texture_array("assets/stbn_vec2.png", layers=64)
        self.random_vec2.label = "texarr_stbn_vec2"
        self.random_vec2.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.gbuffer_lighting["u_albedo"].value = 0
        self.gbuffer_lighting["u_normal"].value = 1
        self.gbuffer_lighting["u_depth"].value = 2
        self.gbuffer_lighting["u_voxel_data"].value = 3
        self.gbuffer_lighting["u_normals"].value = 4
        self.gbuffer_lighting["u_random_vec2"].value = 5
        self.gbuffer_lighting["u_motion_vector"].value = 6
        self.gbuffer_lighting["u_last_frame"].value = 7
        self.gbuffer_lighting["u_last_frame_depth"].value = 8
        self.gbuffer_lighting["u_linear_depth"].value = 9

    @property
    def lighting_texture(self) -> Texture:
        return cast("Texture", self.framebuffers[self.pingpong].color_attachments[0])

    @property
    def last_frame(self) -> Texture:
        return cast("Texture", self.framebuffers[1 - self.pingpong].color_attachments[0])

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        gbuffer: GBuffer,
        voxel_texture: Texture3D,
        frame_counter: int,
        light_pos: Vec3,
        last_frame_depth: Texture3D,
    ) -> None:
        framebuffer = self.framebuffers[self.pingpong]
        ctx = framebuffer.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        framebuffer.clear()
        framebuffer.use()

        self.gbuffer_lighting["uCameraPos"].write(camera.position)
        self.gbuffer_lighting["frame_counter"].value = frame_counter
        self.gbuffer_lighting["uProjection"].write(camera.projection.matrix)
        self.gbuffer_lighting["uView"].write(camera.matrix)
        self.gbuffer_lighting["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.gbuffer_lighting["uInvView"].write(glm.inverse(camera.matrix))
        self.gbuffer_lighting["lightPos"].write(light_pos)
        gbuffer.albedo_texture.use(location=0)
        gbuffer.smooth_normal_texture.use(location=1)
        gbuffer.depth_texture.use(location=2)
        voxel_texture.use(location=3)
        self.stbnormals.use(location=4)
        self.random_vec2.use(location=5)
        gbuffer.motion_vectors.use(location=6)
        self.last_frame.use(location=7)
        last_frame_depth.use(location=8)
        gbuffer.linear_depth.use(location=9)
        self.quad_fs.render(self.gbuffer_lighting)

        self.pingpong = 1 - self.pingpong


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
        self.gbuffer_lighting = GBufferLighting(self, SCREEN_DIMENSIONS)
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
