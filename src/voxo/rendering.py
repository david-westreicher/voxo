from collections.abc import Sequence
from functools import cached_property

import moderngl
import moderngl_window
from moderngl import Framebuffer, Program, Texture
from moderngl_window import geometry
from moderngl_window.scene import Camera
from pyglm import glm

from .constants import GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS
from .objects import Light, Object, Sun, VoxelObject


class GBuffer:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.albedo_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.normal_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.normal_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.motion_vectors = window.ctx.texture(size=size, components=2, dtype="f2")
        # NOTE(david): internally uses GL_DEPTH_COMPONENT24 but we want GL_DEPTH_COMPONENT32F
        self.depth_texture = window.ctx.depth_texture(size=size)
        # NOTE(david): storing linear depth in 32bit float may be unnecessary
        self.linear_depth = window.ctx.texture(size=size, components=1, dtype="f4")
        self.linear_depth.filter = moderngl.NEAREST, moderngl.NEAREST
        self.material_texture = window.ctx.texture(size=size, components=4, dtype="f2")

        self.smooth_normal_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.smooth_normal_texture.filter = moderngl.NEAREST, moderngl.NEAREST

        self.framebuffer = window.ctx.framebuffer(
            color_attachments=[
                self.albedo_texture,
                self.normal_texture,
                self.linear_depth,
                self.material_texture,
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
        self.material_texture.label = f"tex2d_gbuffer_{pingpong}_material"
        self.framebuffer.label = f"framebuffer_gbuffer_{pingpong}"

    def start(self) -> None:
        # Clear depth and linear depth buffers
        self.framebuffer.color_mask = [(val,) * 4 for val in [False, False, True, False, False]]
        self.framebuffer.clear(red=max(GLOBAL_OCCLUDER_DIMENSIONS) * 10.0, depth=1.0)

        ctx = self.framebuffer.ctx
        ctx.enable_only(moderngl.DEPTH_TEST)
        self.framebuffer.color_mask = [(True,) * 4] * len(self.framebuffer.color_attachments)
        self.framebuffer.use()

    def smooth_normals(self, camera: Camera) -> None:
        self.normal_smoother.render(self.normal_texture, self.linear_depth, camera)

    @cached_property
    def textures(self) -> list[Texture]:
        return [
            self.albedo_texture,
            self.normal_texture,
            self.smooth_normal_texture,
            self.depth_texture,
            self.linear_depth,
            self.material_texture,
            self.motion_vectors,
        ]

    @cached_property
    def shaders(self) -> list[Program]:
        return [*self.normal_smoother.shaders]


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

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.program]


class PostProcessing:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.final_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.final_texture.label = "tex2d_postprocessing_final"
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.final_texture])
        self.framebuffer.label = "framebuffer_postprocessing"

        self.postprocessing_program = window.load_program("programs/postprocessing.glsl", defines=GLOBAL_DEFINE)
        self.postprocessing_program.label = "prog_postprocessing"
        self.tonemapping_program = window.load_program("programs/tonemapping.glsl", defines=GLOBAL_DEFINE)
        self.tonemapping_program.label = "prog_tonemapping"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

        self.irradiance_taa = TAA(window, size, "irradiance")
        self.irradiance_taa_2 = TAA(window, size, "irradiance_2")
        self.specular_taa = TAA(window, size, "specular")

        self.bloom = Bloom(window, size)

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        suns: Sequence[Sun],
        irradiance: Texture,
        specular: Texture,
        current_gbuffer: GBuffer,
        last_gbuffer: GBuffer,
        frame_counter: int,
        *,
        camera_moved: bool,
    ) -> None:
        self.irradiance_taa.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=irradiance,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter,
            last_texture=self.irradiance_taa_2.clean_texture,
        )
        self.irradiance_taa_2.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=self.irradiance_taa.clean_texture,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter + 1,
        )
        self.specular_taa.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=specular,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter,
        )

        self.framebuffer.use()

        self.postprocessing_program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.postprocessing_program["uInvView"].write(glm.inverse(camera.matrix))
        if suns and suns[0].visible:
            self.postprocessing_program["sun_direction"].write(suns[0].direction)
        else:
            self.postprocessing_program["sun_direction"].write(glm.vec3(0, -1, 0))
        current_gbuffer.albedo_texture.use(location=0)
        self.irradiance_taa_2.clean_texture.use(location=1)
        self.specular_taa.clean_texture.use(location=2)
        current_gbuffer.depth_texture.use(location=3)
        current_gbuffer.material_texture.use(location=4)
        self.quad.render(self.postprocessing_program)

        self.bloom.render(self.final_texture)
        self.framebuffer.use()
        self.bloom.add_final_bloom(strength=1.0)

    def render_final_tonemapped_texture(self) -> None:
        self.final_texture.use(location=0)
        self.quad.render(self.tonemapping_program)

    @cached_property
    def textures(self) -> list[Texture]:
        return [
            self.final_texture,
            *self.irradiance_taa.textures,
            *self.irradiance_taa_2.textures,
            *self.specular_taa.textures,
            *self.bloom.textures,
        ]

    @cached_property
    def shaders(self) -> list[Program]:
        return [
            self.postprocessing_program,
            self.tonemapping_program,
            *self.irradiance_taa.shaders,
            *self.irradiance_taa_2.shaders,
            *self.specular_taa.shaders,
            *self.bloom.shaders,
        ]


class TAA:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int], name: str) -> None:  # type: ignore[name-defined]
        self.pingpong = 0
        self.textures: list[Texture] = []
        self.framebuffers: list[Framebuffer] = []
        for i in range(2):
            self.textures.append(window.ctx.texture(size=size, components=3, dtype="f2"))
            self.textures[-1].filter = moderngl.LINEAR, moderngl.LINEAR
            self.textures[-1].label = f"tex2d_postprocessing_taa_{name}_{i}"
            self.framebuffers.append(window.ctx.framebuffer(color_attachments=[self.textures[-1]]))
            self.framebuffers[-1].label = f"framebuffer_taa_{name}_{i}"
        self.stbn_scalar = window.load_texture_array("assets/stbn_scalar.png", layers=64)
        self.stbn_scalar.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.program = window.load_program("programs/taa.glsl", defines=GLOBAL_DEFINE)
        self.program.label = f"prog_postprocessing_taa_{name}"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    @property
    def current_framebuffer(self) -> Framebuffer:
        return self.framebuffers[self.pingpong]

    @property
    def last_texture(self) -> Texture:
        return self.textures[1 - self.pingpong]

    @property
    def clean_texture(self) -> Texture:
        return self.textures[self.pingpong]

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        current_texture: Texture,
        motion_vectors: Texture,
        current_depth: Texture,
        last_depth: Texture,
        current_normals: Texture,
        frame_counter: int,
        *,
        camera_moved: bool,
        last_texture: Texture | None = None,
    ) -> None:
        self.pingpong = 1 - self.pingpong
        self.current_framebuffer.use()
        self.program["frame_counter"] = frame_counter
        self.program["u_inv_projection"].write(glm.inverse(camera.projection.matrix))
        self.program["u_inv_view"].write(glm.inverse(camera.matrix))
        self.program["use_history_clamping"] = camera_moved

        if last_texture:
            last_texture.use(location=0)
        else:
            self.last_texture.use(location=0)
        current_texture.use(location=1)
        motion_vectors.use(location=2)
        current_depth.use(location=3)
        current_normals.use(location=4)
        self.stbn_scalar.use(location=5)
        last_depth.use(location=6)
        self.quad.render(self.program)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.program]


class Bloom:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.blurrer = []
        size = (size[0] // 2, size[1] // 2)

        self.exposed_texture = window.ctx.texture(size, components=3, dtype="f2")
        self.exposed_texture.label = "tex_postprocessing_bloom_exposed_texture"
        self.exposed_texture.repeat_x = False
        self.exposed_texture.repeat_y = False
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.exposed_texture])
        self.framebuffer.label = "framebuffer_postprocessing_bloom_exposed_texture"

        self.extract_bloom = window.load_program("programs/extract_bloom.glsl", defines=GLOBAL_DEFINE)
        self.extract_bloom["exposure"] = 2.5
        self.extract_bloom.label = "prog_postprocessing_bloom_extract_bloom"

        while min(size) >= 4:
            size = (size[0] // 2, size[1] // 2)
            self.blurrer.append(Blur(window, size))

        self.upsample_blur = window.load_program("programs/upsample_blur.glsl", defines=GLOBAL_DEFINE)
        self.upsample_blur["strength"] = 1.0
        self.upsample_blur.label = "prog_postprocessing_bloom_upsample_blur"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    def render(self, current_texture: Texture) -> None:
        self.framebuffer.use()
        current_texture.use(location=0)
        self.quad.render(self.extract_bloom)

        current_input_texture = self.exposed_texture
        for blur in self.blurrer:
            blur.render(current_input_texture)
            current_input_texture = blur.current_texture

        ctx = self.exposed_texture.ctx
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        self.upsample_blur["strength"] = 0.7
        for blur in reversed(self.blurrer[:-1]):
            blur.framebuffers[0].use()
            current_input_texture.use(location=0)
            self.quad.render(self.upsample_blur)
            current_input_texture = blur.textures[0]

        self.framebuffer.use()
        current_input_texture.use(location=0)
        self.quad.render(self.upsample_blur)
        ctx.disable(moderngl.BLEND)

    def add_final_bloom(self, strength: float) -> None:
        ctx = self.exposed_texture.ctx
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.exposed_texture.use(location=0)
        self.upsample_blur["strength"] = strength
        self.quad.render(self.upsample_blur)
        ctx.disable(moderngl.BLEND)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.extract_bloom, self.upsample_blur, *[shader for blur in self.blurrer for shader in blur.shaders]]

    @cached_property
    def textures(self) -> list[Texture]:
        return [self.exposed_texture, *[tex for blur in self.blurrer for tex in blur.textures]]


class Blur:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.textures: list[Texture] = []
        self.framebuffers = []
        for i in range(2):
            texture = window.ctx.texture(size, components=3, dtype="f2")
            texture.label = f"tex_postprocessing_blur_{size}_{i}"
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = False
            texture.repeat_y = False
            framebuffer = window.ctx.framebuffer(color_attachments=[texture])
            framebuffer.label = f"framebuffer_postprocessing_blur_{size}_{i}"
            self.textures.append(texture)
            self.framebuffers.append(framebuffer)

        self.blur_vert = window.load_program("programs/blur.glsl", defines=GLOBAL_DEFINE | {"HORIZONTAL": "0"})
        self.blur_vert.label = f"prog_postprocessing_blur_vert_{size}"
        self.blur_horiz = window.load_program("programs/blur.glsl", defines=GLOBAL_DEFINE | {"HORIZONTAL": "1"})
        self.blur_horiz.label = f"prog_postprocessing_blur_horiz_{size}"

        self.quad = geometry.quad_fs(normals=False, uvs=True)

    @property
    def current_texture(self) -> Texture:
        return self.textures[-1]

    def render(self, input_texture: Texture) -> None:
        self.framebuffers[0].use()
        input_texture.use(location=0)
        self.quad.render(self.blur_vert)

        self.framebuffers[1].use()
        self.textures[0].use(location=0)
        self.quad.render(self.blur_horiz)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.blur_vert, self.blur_horiz]


class WireFrameRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog.label = "prof_cube_simple"
        self.prog["color"].value = 1.0, 1.0, 0.0
        self.cube = geometry.cube(size=(1, 1, 1), center=(0.5, 0.5, 0.5))

    def render(self, camera: Camera, objects: Sequence[Object | Light]) -> None:
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
            if object_to_render.geometry is None:
                assert type(object_to_render) is VoxelObject
                self.prog["scale"].write(glm.vec3(object_to_render.model.opengl_dimensions))
                self.cube.render(self.prog)
            else:
                self.prog["scale"].write(glm.vec3(1.0))
                if isinstance(object_to_render, Light):
                    ctx.enable_only(moderngl.CULL_FACE)
                    ctx.cull_face = "front"
                    object_to_render.proxy_object.render(self.prog)
                    ctx.disable(moderngl.CULL_FACE)
                else:
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

    @cached_property
    def textures(self) -> list[Texture]:
        return [tex for buffer in self.buffers for tex in buffer.textures]

    @cached_property
    def shaders(self) -> list[Program]:
        return [shader for buffer in self.buffers for shader in buffer.shaders]

    def swap(self) -> None:
        self.pingpong = 1 - self.pingpong
