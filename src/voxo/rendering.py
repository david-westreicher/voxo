from collections.abc import Sequence
from functools import cached_property

import moderngl
from moderngl import Framebuffer, Program, Texture
from moderngl_window import geometry
from moderngl_window.context.base.window import WindowConfig
from moderngl_window.scene import Camera
from pyglm import glm

from .constants import GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS, USE_TAA
from .objects import Light, Object, Sun, VoxelObject, Water


class GBuffer:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
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
        self.framebuffer.color_mask = [(val,) * 4 for val in [False, False, True, False, False]]  # type:ignore[assignment]
        self.framebuffer.clear(red=max(GLOBAL_OCCLUDER_DIMENSIONS) * 10.0, depth=1.0)
        # Clear albedo
        self.framebuffer.color_mask = [(val,) * 4 for val in [True, False, False, False, False]]  # type:ignore[assignment]
        self.framebuffer.clear(red=0.0, green=0.0, blue=0.0)
        # Clear motion_vector flag
        self.framebuffer.color_mask = [(val,) * 4 for val in [False, False, False, False, True]]  # type:ignore[assignment]
        self.framebuffer.clear(red=-2.0, green=0.0, blue=0.0)

        ctx = self.framebuffer.ctx
        ctx.enable_only(moderngl.DEPTH_TEST)
        self.framebuffer.color_mask = [(True,) * 4] * len(self.framebuffer.color_attachments)  # type:ignore[assignment]
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
    def __init__(self, window: WindowConfig, output_texture: Texture) -> None:
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
    def __init__(self, window: WindowConfig, size: tuple[int, int], skybox: Texture) -> None:
        self.sky_texture = skybox

        self.final_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.final_texture.label = "tex2d_postprocessing_final"
        self.final_texture.repeat_x = False
        self.final_texture.repeat_y = False
        self.final_framebuffer = window.ctx.framebuffer(color_attachments=[self.final_texture])
        self.final_framebuffer.label = "framebuffer_postprocessing_final"

        self.postprocessing_program = window.load_program("programs/postprocessing.glsl", defines=GLOBAL_DEFINE)
        self.postprocessing_program.label = "prog_postprocessing"
        self.tonemapping_program = window.load_program("programs/tonemapping.glsl", defines=GLOBAL_DEFINE)
        self.tonemapping_program.label = "prog_tonemapping"
        self.copy_program = window.load_program("programs/copy.glsl", defines=GLOBAL_DEFINE)
        self.copy_program.label = "prog_copy"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

        self.bloom = Bloom(window, size)
        self.taa = TAA(window, size)

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        motion_vectors: Texture,
        suns: Sequence[Sun],
        light_texture: Texture,
        depth_texture: Texture,
        linear_depth_texture: Texture,
        prev_linear_depth_texture: Texture,
        reflectivity_texture: Texture,
    ) -> None:
        sun_direction = suns[0].direction if suns and suns[0].visible else glm.vec3(0, -1, 0)
        sun_color = suns[0].color if suns and suns[0].visible else glm.vec3(0, -1, 0)

        self.final_framebuffer.use()
        self.postprocessing_program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.postprocessing_program["uInvView"].write(glm.inverse(camera.matrix))
        self.postprocessing_program["sun_direction"].write(sun_direction)
        self.postprocessing_program["sun_color"].write(sun_color)
        light_texture.use(location=0)
        depth_texture.use(location=1)
        self.sky_texture.use(location=2)
        self.quad.render(self.postprocessing_program)
        if USE_TAA:
            self.taa.render(
                self.final_texture,
                motion_vectors,
                linear_depth_texture,
                prev_linear_depth_texture,
                reflectivity_texture,
            )
            self.bloom.render(self.taa.clean_texture)
        else:
            self.bloom.render(self.final_texture)

        self.final_framebuffer.use()
        if USE_TAA:
            self.taa.clean_texture.use(location=0)
            self.quad.render(self.copy_program)
        self.bloom.add_final_bloom(strength=1.0)

    def render_final_tonemapped_texture(self) -> None:
        self.final_texture.use(location=0)
        self.quad.render(self.tonemapping_program)

    @cached_property
    def textures(self) -> list[Texture]:
        return [
            self.final_texture,
            self.sky_texture,
            *self.bloom.textures,
            *self.taa.textures,
        ]

    @cached_property
    def shaders(self) -> list[Program]:
        return [
            self.postprocessing_program,
            self.tonemapping_program,
            *self.bloom.shaders,
            *self.taa.shaders,
        ]


class Denoiser:
    def __init__(self, window: WindowConfig, size: tuple[int, int], name: str) -> None:
        self.pingpong = 0
        self.textures: list[Texture] = []
        self.framebuffers: list[Framebuffer] = []
        for i in range(2):
            self.textures.append(window.ctx.texture(size=size, components=3, dtype="f2"))
            self.textures[-1].filter = moderngl.LINEAR, moderngl.LINEAR
            self.textures[-1].label = f"tex2d_{name}_{i}"
            self.framebuffers.append(window.ctx.framebuffer(color_attachments=[self.textures[-1]]))
            self.framebuffers[-1].label = f"framebuffer_{name}_{i}"
        self.stbn_scalar = window.load_texture_array("assets/stbn_scalar.png", layers=64)
        self.stbn_scalar.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.program = window.load_program("programs/denoise.glsl", defines=GLOBAL_DEFINE)
        self.program.label = f"prog_{name}"
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


class ATrousDenoiser:
    def __init__(self, window: WindowConfig, size: tuple[int, int], name: str) -> None:
        self.pingpong = 0
        self.textures: list[Texture] = []
        self.framebuffers: list[Framebuffer] = []
        for i in range(2):
            self.textures.append(window.ctx.texture(size=size, components=3, dtype="f2"))
            self.textures[-1].filter = moderngl.LINEAR, moderngl.LINEAR
            self.textures[-1].label = f"tex2d_{name}_{i}"
            self.textures[-1].repeat_x = False
            self.textures[-1].repeat_y = False
            self.framebuffers.append(window.ctx.framebuffer(color_attachments=[self.textures[-1]]))
            self.framebuffers[-1].label = f"framebuffer_{name}_{i}"
        self.program = window.load_program("programs/atrous.glsl", defines=GLOBAL_DEFINE)
        self.program.label = f"prog_{name}"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    @property
    def current_framebuffer(self) -> Framebuffer:
        return self.framebuffers[self.pingpong]

    @property
    def clean_texture(self) -> Texture:
        return self.textures[self.pingpong]

    def render(
        self,
        current_texture: Texture,
        current_depth: Texture,
        current_normals: Texture,
        *,
        step_size: float = 1.0,
    ) -> None:
        self.pingpong = 1 - self.pingpong
        self.current_framebuffer.use()
        self.program["step_size"] = step_size

        current_texture.use(location=0)
        current_depth.use(location=1)
        current_normals.use(location=2)
        self.quad.render(self.program)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.program]


class TAA:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
        self.pingpong = 0
        self.textures = []
        self.framebuffers = []
        for i in range(2):
            texture = window.ctx.texture(size, components=3, dtype="f2")
            texture.label = f"texture2d_taa_{i}"
            texture.repeat_x = False
            texture.repeat_y = False
            framebuffer = window.ctx.framebuffer(color_attachments=[texture])
            self.textures.append(texture)
            self.framebuffers.append(framebuffer)

        self.taa_program = window.load_program("programs/taa.glsl", defines=GLOBAL_DEFINE)
        self.taa_program.label = "program_taa"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    @property
    def last_texture(self) -> Texture:
        return self.textures[1 - self.pingpong]

    @property
    def clean_texture(self) -> Texture:
        return self.textures[self.pingpong]

    @property
    def current_framebuffer(self) -> Framebuffer:
        return self.framebuffers[self.pingpong]

    def render(
        self,
        image: Texture,
        motion_vectors: Texture,
        linear_depth_texture: Texture,
        prev_linear_depth_texture: Texture,
        reflectivity_texture: Texture,
    ) -> None:
        self.pingpong = 1 - self.pingpong

        image.use(location=0)
        self.last_texture.use(location=1)
        motion_vectors.use(location=2)
        linear_depth_texture.use(location=3)
        prev_linear_depth_texture.use(location=4)
        reflectivity_texture.use(location=5)

        self.framebuffers[self.pingpong].use()
        self.quad.render(self.taa_program)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.taa_program]


class Bloom:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
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

        while min(size) > 8:
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
        ctx.blend_equation = moderngl.FUNC_ADD  # type:ignore[assignment]
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
        ctx.blend_equation = moderngl.FUNC_ADD  # type:ignore[assignment]
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
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
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


class WaterRenderer:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
        self.water_normal_1 = window.load_texture_2d("assets/water_01_normal.jpg")
        self.water_normal_1.label = "texture2d_water_01_normal"
        self.water_normal_2 = window.load_texture_2d("assets/water_02_normal.jpg")
        self.water_normal_2.label = "texture2d_water_02_normal"
        self.albedo_copy = window.ctx.texture(size, components=3, dtype="f2")
        self.albedo_copy.label = "texture2d_albedo_copy"
        self.albedo_copy.repeat_x = False
        self.albedo_copy.repeat_y = False
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.albedo_copy])
        self.framebuffer.label = "framebuffer_water_albedo_copy"

        self.prog = window.load_program("programs/water.glsl", defines=GLOBAL_DEFINE)
        self.prog.label = "prog_water"
        self.copy_program = window.load_program("programs/copy.glsl", defines=GLOBAL_DEFINE)
        self.copy_program.label = "prog_copy"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    def copy_albedo(self, gbuffer: GBuffer) -> None:
        self.framebuffer.use()
        gbuffer.albedo_texture.use(location=0)
        self.quad.render(self.copy_program)

    def render(self, camera: Camera, gbuffer: GBuffer, waters: list[Water], frame_counter: int) -> None:
        ctx = self.prog.ctx

        ctx.enable_only(moderngl.DEPTH_TEST)
        self.prog["m_proj"].write(camera.projection.matrix)
        self.prog["m_camera"].write(camera.matrix)
        self.prog["u_camera_pos"].write(camera.position)
        self.prog["u_frame_counter"] = frame_counter

        gbuffer.linear_depth.use(location=0)
        self.albedo_copy.use(location=1)
        self.water_normal_1.use(location=2)
        self.water_normal_2.use(location=3)
        for water in waters:
            if not water.visible:
                continue
            self.prog["m_model"].write(water.transform)
            self.prog["u_color"].write(water.color)
            assert water.geometry
            water.geometry.render(self.prog)
        ctx.disable(moderngl.DEPTH_TEST)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.prog]

    @cached_property
    def textures(self) -> list[Texture]:
        return [self.water_normal_1, self.water_normal_2]


class WireFrameRenderer:
    def __init__(self, window: WindowConfig) -> None:
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog.label = "prog_cube_simple"
        self.prog["color"].value = 1.0, 1.0, 0.0
        self.cube = geometry.cube(size=(1, 1, 1), center=(0.5, 0.5, 0.5))

    def render(self, camera: Camera, objects: Sequence[Object | Light | Water]) -> None:
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
                    self.prog["m_model"].write(object_to_render.proxy_transform)
                    object_to_render.proxy_geometry.render(self.prog)
                    ctx.disable(moderngl.CULL_FACE)
                else:
                    object_to_render.geometry.render(self.prog)
        ctx.wireframe = False


class GBufferPingPong:
    def __init__(self, window: WindowConfig, dimensions: tuple[int, int]) -> None:
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
