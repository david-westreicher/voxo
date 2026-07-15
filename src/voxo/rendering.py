from collections.abc import Sequence

import moderngl
import moderngl_window
from moderngl import Texture
from moderngl_window import geometry
from moderngl_window.scene import Camera
from pyglm import glm

from .constants import GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS
from .objects import Object, Sun

GL_RGB10_A2 = 0x8059
GL_DEPTH_COMPONENT32F = 0x8CAC


class GBuffer:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.albedo_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.normal_texture = window.ctx.texture(size=size, components=3, internal_format=GL_RGB10_A2)
        self.smooth_normal_texture = window.ctx.texture(size=size, components=3, internal_format=GL_RGB10_A2)
        self.motion_vectors = window.ctx.texture(size=size, components=2, dtype="f2")
        # NOTE(david): internally uses GL_DEPTH_COMPONENT24 but we want GL_DEPTH_COMPONENT32F
        self.depth_texture = window.ctx.depth_texture(size=size)
        self.linear_depth = window.ctx.texture(size=size, components=1, dtype="f2")

        self.framebuffer = window.ctx.framebuffer(
            color_attachments=[
                self.albedo_texture,
                self.normal_texture,
                self.linear_depth,
                self.motion_vectors,
            ],
            depth_attachment=self.depth_texture,
        )
        self.normal_smoother = SmoothNormals(window, self.smooth_normal_texture)

    def label(self, pingpong: int = 0) -> None:
        self.albedo_texture.label = f"tex2d_gbuffer_{pingpong}_albedo"
        self.normal_texture.label = f"tex2d_gbuffer_{pingpong}_normal"
        self.smooth_normal_texture.label = f"tex2d_gbuffer_{pingpong}_smooth_normal"
        self.motion_vectors.label = f"tex2d_gbuffer_{pingpong}_motion_vectors"
        self.depth_texture.label = f"tex2d_gbuffer_{pingpong}_depth"
        self.linear_depth.label = f"tex2d_gbuffer_{pingpong}_linear_depth"
        self.framebuffer.label = f"framebuffer_gbuffer_{pingpong}"

    def start(self) -> None:
        self.framebuffer.color_mask = (
            [(False,) * 4] * (len(self.framebuffer.color_attachments) - 2) + [(True,) * 4] + [(False,) * 4]
        )
        self.framebuffer.clear(red=max(GLOBAL_OCCLUDER_DIMENSIONS) * 10.0, depth=1.0)

        ctx = self.framebuffer.ctx
        ctx.enable_only(moderngl.DEPTH_TEST)
        self.framebuffer.color_mask = [(True,) * 4] * len(self.framebuffer.color_attachments)
        self.framebuffer.use()
        self.albedo_texture.use(location=0)
        self.normal_texture.use(location=1)
        self.linear_depth.use(location=2)
        self.motion_vectors.use(location=3)

    def smooth_normals(self, camera: Camera) -> None:
        self.normal_smoother.render(self.normal_texture, self.linear_depth, camera)


class SmoothNormals:
    def __init__(self, window: moderngl_window.WindowConfig, output_texture: Texture) -> None:  # type: ignore[name-defined]
        self.program = window.load_program("programs/smooth_normals.glsl")
        self.program.label = "prog_smooth_normals"
        self.program["input_texture"].value = 0
        self.program["depth_texture"].value = 1
        self.quad = geometry.quad_fs(normals=False, uvs=True)
        self.framebuffer = window.ctx.framebuffer(color_attachments=[output_texture])
        self.framebuffer.label = "framebuffer_smooth_normals"

    def render(self, input_texture: Texture, depth_texture: Texture, camera: Camera) -> None:
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.program["uInvView"].write(glm.inverse(camera.matrix))
        ctx = self.framebuffer.ctx
        ctx.disable(moderngl.DEPTH_TEST)

        self.framebuffer.use()
        input_texture.use(location=0)
        depth_texture.use(location=1)
        self.quad.render(self.program)


class GBufferDebug:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_debug = window.load_program("programs/gbuffer_debug.glsl")
        self.gbuffer_debug.label = "prog_gbuffer_debug"
        self.gbuffer_debug["u_albedo"].value = 0
        self.gbuffer_debug["u_normal"].value = 1
        self.gbuffer_debug["u_depth"].value = 2
        self.gbuffer_debug["u_motion_vectors"].value = 3
        self.gbuffer_debug["u_lighting"].value = 4

    def render(self, gbuffer: GBuffer, final_hdr_texture: Texture, *, debug: bool) -> None:
        gbuffer.albedo_texture.use(location=0)
        gbuffer.smooth_normal_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        gbuffer.motion_vectors.use(location=3)
        final_hdr_texture.use(location=4)
        self.gbuffer_debug["full"].value = not debug
        self.quad_fs.render(self.gbuffer_debug)


class PostProcessing:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.final_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.final_texture.label = "tex2d_postprocessing_final"
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.final_texture])
        self.framebuffer.label = "framebuffer_postprocessing"

        self.program = window.load_program("programs/postprocessing.glsl", defines=GLOBAL_DEFINE)
        self.program.label = "prog_postprocessing"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    def render(self, camera: Camera, suns: Sequence[Sun], albedo: Texture, irradiance: Texture, depth: Texture) -> None:
        self.framebuffer.use()

        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.program["uInvView"].write(glm.inverse(camera.matrix))
        if suns:
            self.program["sun_direction"].write(suns[0].direction)
        else:
            self.program["sun_direction"].write(glm.vec3(0, -1, 0))
        albedo.use(location=0)
        irradiance.use(location=1)
        depth.use(location=2)
        self.quad.render(self.program)


class WireFrameRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog.label = "prof_cube_simple"
        self.prog["color"].value = 1.0, 1.0, 0.0

    def render(self, camera: Camera, objects: Sequence[Object]) -> None:
        ctx = self.prog.ctx
        ctx.enable_only(moderngl.CULL_FACE)
        ctx.wireframe = True
        for object_to_render in objects:
            if hasattr(object_to_render, "color"):
                self.prog["color"].write(glm.normalize(object_to_render.color))
            else:
                self.prog["color"].value = 1.0, 1.0, 0.0
            self.prog["m_proj"].write(camera.projection.matrix)
            self.prog["m_model"].write(object_to_render.transform)
            self.prog["m_camera"].write(camera.matrix)
            object_to_render.geometry.render(self.prog)
        ctx.wireframe = False


class GBufferPingPong:
    def __init__(self, window: moderngl_window.WindowConfig, dimensions: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.buffers = [GBuffer(window, dimensions) for _ in range(2)]
        for i, gbuffer in enumerate(self.buffers):
            gbuffer.label(i)
        self.pingpong = 0

    @property
    def current(self) -> GBuffer:
        return self.buffers[self.pingpong]

    @property
    def last(self) -> GBuffer:
        return self.buffers[1 - self.pingpong]

    def swap(self) -> None:
        self.pingpong = 1 - self.pingpong
