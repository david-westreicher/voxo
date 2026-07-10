from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import moderngl
import moderngl_window
from moderngl import Program
from moderngl_window import geometry
from moderngl_window.context.base import KeyModifiers
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene import Camera
from moderngl_window.scene.camera import KeyboardCamera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812

from .model import parse_model

MODEL_PATH = Path("./resources/models/truck.txt")
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
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        data: bytes,
        dimensions: tuple[int, int, int],
        palette: bytes,
    ) -> None:
        self.program: Program = window.load_program("programs/raytrace_voxels.glsl")
        self.cube = Object(geometry.cube(size=(1, 1, 1)))
        self.cube.translation = glm.translate(glm.vec3(0.0))
        self.cube.scale = glm.scale(glm.vec3(dimensions))

        self.voxel_texture = window.ctx.texture3d(dimensions, data=data, components=1, alignment=1, dtype="u1")
        self.voxel_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.voxel_texture.repeat_x = False
        self.voxel_texture.repeat_y = False
        self.voxel_texture.repeat_z = False

        self.palette_texture = window.ctx.texture((len(palette) // 3, 1), data=palette, components=3, dtype="f1")
        self.palette_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.palette_texture.repeat_x = False
        self.palette_texture.repeat_y = False
        self.palette_texture.repeat_z = False

    def render(self, camera: Camera, time: float) -> None:
        ctx = self.voxel_texture.ctx
        self.program["m_proj"].write(camera.projection.matrix)  # type:ignore[union-attr]
        self.program["m_model"].write(self.cube.modelview)  # type:ignore[union-attr]
        self.program["m_camera"].write(camera.matrix)  # type:ignore[union-attr]
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))  # type:ignore[union-attr]
        self.program["uInvView"].write(glm.inverse(camera.matrix))  # type:ignore[union-attr]
        self.program["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.program["u_voxel_data"].value = 0  # type:ignore[union-attr]
        self.program["u_palette_data"].value = 1  # type:ignore[union-attr]
        self.program["time"].value = time  # type:ignore[union-attr]

        self.voxel_texture.use(location=0)
        self.palette_texture.use(location=1)
        self.cube.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


class RenderIntoTexture:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.framebuffer_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.framebuffer_texture])
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.program = window.load_program("programs/tonemapping.glsl")
        self.program["u_texture"].value = 0

    def start(self) -> None:
        self.framebuffer.clear(0.0, 0.0, 0.0, 0.0)
        self.framebuffer.use()

    def render(self) -> None:
        self.framebuffer_texture.use(location=0)
        self.quad_fs.render(self.program)


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
        ctx = self.framebuffer.ctx
        # TODO(david): Can i set the self.framebuffer.color_mask here to just clear depth?
        self.framebuffer.depth_mask = True
        ctx.enable_only(moderngl.DEPTH_TEST)
        ctx.depth_func = "1"
        self.framebuffer.clear(depth=1.0)
        self.framebuffer.use()
        self.framebuffer.depth_mask = True
        self.albedo_texture.use(location=0)
        self.normal_texture.use(location=1)


class GBufferLighting:
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        size: tuple[int, int],
        gbuffer: GBuffer,
        voxel_texture: moderngl.Texture,
    ) -> None:
        self.lighting_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.lighting_texture])
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_lighting = window.load_program("programs/gbuffer_lighting.glsl")
        self.gbuffer_lighting["u_albedo"].value = 0
        self.gbuffer_lighting["u_normal"].value = 1
        self.gbuffer_lighting["u_depth"].value = 2
        self.gbuffer_lighting["u_voxel_data"].value = 3  # type:ignore[union-attr]
        self.gbuffer = gbuffer
        self.voxel_texture = voxel_texture

    def render(self, camera: Camera, time: float) -> None:
        ctx = self.framebuffer.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        self.framebuffer.clear()
        self.framebuffer.use()

        self.gbuffer_lighting["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.gbuffer_lighting["time"].value = time  # type:ignore[union-attr]
        self.gbuffer_lighting["u_inv_proj_view"].write(glm.inverse(camera.projection.matrix @ camera.matrix))
        self.gbuffer.albedo_texture.use(location=0)
        self.gbuffer.normal_texture.use(location=1)
        self.gbuffer.depth_texture.use(location=2)
        self.voxel_texture.use(location=3)
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

    def render(self) -> None:
        self.gbuffer.albedo_texture.use(location=0)
        self.gbuffer.normal_texture.use(location=1)
        self.gbuffer.depth_texture.use(location=2)
        self.gbuffer_lighting.use(location=3)
        self.quad_fs.render(self.gbuffer_debug)


class SkyRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.program = window.load_program("programs/sky_renderer.glsl")

    def render(self, camera: Camera) -> None:
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.program["uInvView"].write(glm.inverse(camera.matrix))
        self.quad_fs.render(self.program)


class WireFrameBox:
    def __init__(self, window: moderngl_window.WindowConfig, dimensions: tuple[float, float, float]) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog["color"].value = 1.0, 1.0, 0.0, 1.0

        self.cube = Object(geometry.cube(size=(1, 1, 1)))
        self.cube.translation = glm.translate(glm.vec3(0.0))
        self.cube.scale = glm.scale(glm.vec3(dimensions))

    def render(self, camera: Camera) -> None:
        ctx = self.prog.ctx
        ctx.enable_only(moderngl.CULL_FACE)
        ctx.wireframe = True
        self.prog["m_proj"].write(camera.projection.matrix)
        self.prog["m_model"].write(self.cube.modelview)
        self.prog["m_camera"].write(camera.matrix)
        self.cube.geometry.render(self.prog)
        ctx.wireframe = False


class Box:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/test_frag_depth.glsl")
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    def render(self) -> None:
        self.quad.render(self.prog)


class CubeSimple(CameraWindow):
    gl_version = (4, 6)
    window_size = SCREEN_DIMENSIONS
    title = "voxo"
    resource_dir = Path("resources").resolve()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.time = 0

        model = parse_model(MODEL_PATH)
        print(model.dimensions)
        self.voxel_renderer = VoxelRenderer(
            self,
            model.generate_voxel_data(),
            model.opengl_dimensions,
            model.generate_palette_data(),
        )
        self.gbuffer = GBuffer(self, SCREEN_DIMENSIONS)
        self.gbuffer_lighting = GBufferLighting(
            self,
            SCREEN_DIMENSIONS,
            self.gbuffer,
            self.voxel_renderer.voxel_texture,
        )
        self.gbuffer_debug = GBufferDebug(self, self.gbuffer, self.gbuffer_lighting.lighting_texture)
        self.sky_renderer = SkyRenderer(self)
        self.wireframe_box = WireFrameBox(self, model.opengl_dimensions)
        self.box = Box(self)

    def on_render(self, time: float, frametime: float) -> None:  # noqa: ARG002
        self.time += 1

        # Render into HDR framebuffer
        self.gbuffer.start()
        self.voxel_renderer.render(self.camera, self.time)

        self.gbuffer_lighting.render(self.camera, self.time)

        # Render framebuffer onto screen
        self.ctx.screen.clear(0.1, 0.1, 0.1, 1.0)
        self.ctx.screen.use()
        self.gbuffer_debug.render()
        self.wireframe_box.render(self.camera)
