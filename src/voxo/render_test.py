from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import moderngl
import moderngl_window
from moderngl import Context, Program, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.context.base import KeyModifiers, WindowConfig
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene import Camera
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812
from pyglm.glm import quat as Quat  # noqa: N812
from pyglm.glm import vec3 as Vec3  # noqa: N812

from .model import Model, parse_model

MODEL_PATH = Path("./resources/models/truck_plane.txt")
DWARF = Path("./resources/models/dwarf.txt")
SCREEN_DIMENSIONS = (1920, 1080)
ASPECT_RATIO = SCREEN_DIMENSIONS[0] / SCREEN_DIMENSIONS[1]
GL_RGB10_A2 = 0x8059
GL_DEPTH_COMPONENT32F = 0x8CAC


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


@dataclass
class Object:
    geometry: VAO
    rotation: Quat = field(default=glm.quat())
    translation: Vec3 = field(default=glm.vec3(0.0))
    scale: Vec3 = field(default=glm.vec3(1.0))

    @property
    def transform(self) -> Mat4:
        return cast("Mat4", glm.translate(self.translation) @ glm.mat4_cast(self.rotation) @ glm.scale(self.scale))

    def rotate(self, angle: float, axis: Vec3) -> None:
        self.rotation = cast("Quat", glm.rotate(self.rotation, angle, axis))


@dataclass(kw_only=True)
class VoxelObject(Object):
    model: Model
    geometry: VAO = field(default_factory=lambda: geometry.cube(size=(1, 1, 1)))
    _voxel_texture: Texture3D | None = None
    _palette_texture: Texture | None = None

    def __post_init__(self) -> None:
        self.geometry = geometry.cube(size=self.model.opengl_dimensions)

    def upload_to_gpu(self, ctx: Context) -> None:
        self._voxel_texture = ctx.texture3d(
            self.model.opengl_dimensions,
            data=self.model.generate_voxel_data(),
            components=1,
            alignment=1,
            dtype="u1",
        )
        self._voxel_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._voxel_texture.repeat_x = False
        self._voxel_texture.repeat_y = False
        self._voxel_texture.repeat_z = False

        palette = self.model.generate_palette_data()
        self._palette_texture = ctx.texture((len(palette) // 3, 1), data=palette, components=3, dtype="f1")
        self._palette_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._palette_texture.repeat_x = False
        self._palette_texture.repeat_y = False

    @property
    def voxel_texture(self) -> Texture3D:
        assert self._voxel_texture
        return self._voxel_texture

    @property
    def palette_texture(self) -> Texture:
        assert self._palette_texture
        return self._palette_texture


class VoxelRenderer:
    def __init__(self, window: WindowConfig) -> None:
        self.program: Program = window.load_program("programs/gbuffer_create.glsl")

    def render(self, camera: Camera, voxel_object: VoxelObject) -> None:
        ctx = voxel_object.voxel_texture.ctx
        self.program["m_proj"].write(camera.projection.matrix)  # type:ignore[union-attr]
        self.program["m_model"].write(voxel_object.transform)  # type:ignore[union-attr]
        self.program["m_model_inverse"].write(glm.inverse(voxel_object.transform))  # type:ignore[union-attr]
        self.program["m_camera"].write(camera.matrix)  # type:ignore[union-attr]
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))  # type:ignore[union-attr]
        self.program["uInvView"].write(glm.inverse(camera.matrix))  # type:ignore[union-attr]
        self.program["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.program["u_voxel_data"].value = 0  # type:ignore[union-attr]
        self.program["u_palette_data"].value = 1  # type:ignore[union-attr]

        voxel_object.voxel_texture.use(location=0)
        voxel_object.palette_texture.use(location=1)
        voxel_object.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


class GBuffer:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.albedo_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.normal_texture = window.ctx.texture(size=size, components=3, internal_format=GL_RGB10_A2)
        self.depth_texture = window.ctx.depth_texture(size=size)
        # NOTE(david): internally uses GL_DEPTH_COMPONENT24 but we want GL_DEPTH_COMPONENT32F

        self.framebuffer = window.ctx.framebuffer(
            color_attachments=[
                self.albedo_texture,
                self.normal_texture,
            ],
            depth_attachment=self.depth_texture,
        )

    def start(self) -> None:
        # clear depth buffer
        self.framebuffer.color_mask = [(False,) * 4] * 2
        self.framebuffer.clear(depth=1.0)

        ctx = self.framebuffer.ctx
        ctx.enable_only(moderngl.DEPTH_TEST)
        self.framebuffer.color_mask = [(True,) * 4] * 2
        self.framebuffer.use()
        self.albedo_texture.use(location=0)
        self.normal_texture.use(location=1)


class GBufferLighting:
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        size: tuple[int, int],
    ) -> None:
        self.lighting_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.lighting_texture])
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_lighting = window.load_program("programs/gbuffer_lighting.glsl")
        self.stbnormals = window.load_texture_array("assets/stbn_cosine_normals.png", layers=64)
        self.stbnormals.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.gbuffer_lighting["u_albedo"].value = 0
        self.gbuffer_lighting["u_normal"].value = 1
        self.gbuffer_lighting["u_depth"].value = 2
        self.gbuffer_lighting["u_voxel_data"].value = 3
        self.gbuffer_lighting["u_normals"].value = 4

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        gbuffer: GBuffer,
        voxel_texture: Texture3D,
        time: float,
        light_pos: Vec3,
        voxel_model_view: Mat4,
    ) -> None:
        ctx = self.framebuffer.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        self.framebuffer.clear()
        self.framebuffer.use()

        self.gbuffer_lighting["uCameraPos"].write(camera.position)
        self.gbuffer_lighting["time"].value = time * 50.0
        self.gbuffer_lighting["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.gbuffer_lighting["uInvView"].write(glm.inverse(camera.matrix))
        # TODO(david): Local transformation can be removed as soon we have a global occluder (without transform)
        self.gbuffer_lighting["m_model_inverse"].write(glm.inverse(voxel_model_view))
        self.gbuffer_lighting["lightPos"].write(light_pos)
        gbuffer.albedo_texture.use(location=0)
        gbuffer.normal_texture.use(location=1)
        gbuffer.depth_texture.use(location=2)
        voxel_texture.use(location=3)
        self.stbnormals.use(location=4)
        self.quad_fs.render(self.gbuffer_lighting)


class GBufferDebug:
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        gbuffer: GBuffer,
        gbuffer_lighting_texture: moderngl.Texture,
    ) -> None:
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_debug = window.load_program("programs/gbuffer_debug.glsl")
        self.gbuffer_debug["u_albedo"].value = 0
        self.gbuffer_debug["u_normal"].value = 1
        self.gbuffer_debug["u_depth"].value = 2
        self.gbuffer_debug["u_lighting"].value = 3
        self.gbuffer = gbuffer
        self.gbuffer_lighting = gbuffer_lighting_texture

    def render(self, *, debug: bool) -> None:
        self.gbuffer.albedo_texture.use(location=0)
        self.gbuffer.normal_texture.use(location=1)
        self.gbuffer.depth_texture.use(location=2)
        self.gbuffer_lighting.use(location=3)
        self.gbuffer_debug["full"].value = not debug
        self.quad_fs.render(self.gbuffer_debug)


class WireFrameRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog["color"].value = 1.0, 1.0, 0.0, 1.0

    def render(self, camera: Camera, objects: Sequence[Object]) -> None:
        ctx = self.prog.ctx
        ctx.enable_only(moderngl.CULL_FACE)
        ctx.wireframe = True
        for object_to_render in objects:
            self.prog["m_proj"].write(camera.projection.matrix)
            self.prog["m_model"].write(object_to_render.transform)
            self.prog["m_camera"].write(camera.matrix)
            object_to_render.geometry.render(self.prog)
        ctx.wireframe = False


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
        self.debug = False

        self.voxel_objects = [VoxelObject(model=parse_model(MODEL_PATH)), VoxelObject(model=parse_model(DWARF))]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(self.ctx)
        self.voxel_renderer = VoxelRenderer(self)
        self.light = Object(geometry.sphere(2))
        self.gbuffer = GBuffer(self, SCREEN_DIMENSIONS)
        self.gbuffer_lighting = GBufferLighting(self, SCREEN_DIMENSIONS)
        self.gbuffer_debug = GBufferDebug(self, self.gbuffer, self.gbuffer_lighting.lighting_texture)
        self.wireframe_box = WireFrameRenderer(self)

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_RELEASE and key == keys.B:
            self.debug = not self.debug

    def on_render(self, time: float, frametime: float) -> None:  # noqa: ARG002
        self.time = time
        if self.timer.is_running:
            self.voxel_objects[0].rotate(0.001, glm.vec3(0, 1, 0))
            self.voxel_objects[1].translation = glm.vec3(0, 40, 0)
            self.voxel_objects[1].rotate(-0.001, glm.vec3(0, 1, 0))
            self.light.translation = glm.rotateY(glm.vec3(40, 0, 0), time) + glm.vec3(0, 40, 0)

        # Render into HDR framebuffer
        self.gbuffer.start()
        for voxel_object in self.voxel_objects:
            self.voxel_renderer.render(self.camera, voxel_object)

        # Compute lighting
        self.gbuffer_lighting.render(
            self.camera,
            self.gbuffer,
            self.voxel_objects[0].voxel_texture,
            self.time,
            self.light.translation,
            self.voxel_objects[0].transform,
        )

        # Render framebuffer onto screen
        self.ctx.screen.clear(0.1, 0.1, 0.1, 1.0)
        self.ctx.screen.use()
        self.gbuffer_debug.render(debug=self.debug)
        if not self.debug:
            self.wireframe_box.render(self.camera, self.voxel_objects)
            self.wireframe_box.render(self.camera, [self.light])
