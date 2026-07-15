from collections.abc import Sequence

import moderngl
from moderngl import ComputeShader, Program, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.context.base import WindowConfig
from moderngl_window.scene import Camera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812

from .constants import GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS
from .objects import Light, Object, Sun, VoxelObject
from .rendering import GBuffer


class GlobalOccluder:
    def __init__(self, window: WindowConfig, dimensions: tuple[int, int, int]) -> None:
        self.dimensions = dimensions
        self.blitter: ComputeShader = window.load_compute_shader("programs/blitter.glsl")
        self.blitter["voxel_texture"].value = 0
        self.blitter.label = "prog_blitter"

        self.clearer: ComputeShader = window.load_compute_shader("programs/clearer.glsl")
        self.clearer.label = "prog_clearer"

        self.mipmapper: ComputeShader = window.load_compute_shader("programs/occluder_mipmapper.glsl")
        self.mipmapper.label = "prog_occluder_mipmapper"

        self.debug_shader: Program = window.load_program("programs/debug_occluder.glsl", defines=GLOBAL_DEFINE)
        self.debug_shader["occluder_texture"].value = 0
        self.debug_shader.label = "prog_debug_occluder"
        self.debug_quad = geometry.quad_fs(normals=False, uvs=True)

        self.occluder_texture = window.ctx.texture3d(
            size=dimensions,
            data=None,
            components=1,
            alignment=1,
            dtype="u1",
            create_mip_maps=True,
        )
        self.occluder_texture.filter = moderngl.NEAREST_MIPMAP_NEAREST, moderngl.NEAREST
        self.occluder_texture.label = "tex3d_global_occluder"

        self.occluder_volume = Object(geometry=geometry.cube(size=dimensions))
        self.occluder_volume.translation = glm.vec3(dimensions) * 0.5

    def blit_object(self, voxel_object: VoxelObject) -> None:
        # TODO(david): only blit to affected bounding box
        self.blitter["obj_dimensions"].write(glm.ivec3(voxel_object.model.opengl_dimensions))
        self.blitter["obj_transform_inv"].write(glm.inverse(voxel_object.transform))
        voxel_object.voxel_texture.use(location=0)
        self.occluder_texture.bind_to_image(1, read=False, write=True, level=0)
        self.blitter.run(
            GLOBAL_OCCLUDER_DIMENSIONS[0] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[1] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[2] // 8,
        )

    def clear(self) -> None:
        # TODO(david): only clear affected bounding box
        # TODO(david): We should use glClearTexImage/glClearTexSubImage but it's not supported in moderngl
        self.occluder_texture.bind_to_image(0, read=False, write=True, level=0)
        self.clearer.run(
            GLOBAL_OCCLUDER_DIMENSIONS[0] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[1] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[2] // 8,
        )

    def update_mipmaps(self) -> None:
        destination_dimensions = glm.ivec3(GLOBAL_OCCLUDER_DIMENSIONS) // 2
        dst_mip = 1
        while glm.min(destination_dimensions) > 0:
            self.occluder_texture.bind_to_image(0, read=True, write=False, level=dst_mip - 1)
            self.occluder_texture.bind_to_image(1, read=False, write=True, level=dst_mip)
            self.mipmapper.run(
                (destination_dimensions[0] + 7) // 8,
                (destination_dimensions[1] + 7) // 8,
                (destination_dimensions[2] + 7) // 8,
            )
            destination_dimensions //= 2
            dst_mip += 1
            self.occluder_texture.ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def render_debug(self, camera: Camera) -> None:
        self.occluder_texture.use(location=0)
        self.debug_shader["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.debug_shader["uInvView"].write(glm.inverse(camera.matrix))
        self.debug_quad.render(self.debug_shader)


class VoxelRenderer:
    def __init__(self, window: WindowConfig) -> None:
        self.program: Program = window.load_program("programs/gbuffer_create.glsl", defines=GLOBAL_DEFINE)
        self.program.label = "prog_gbuffer_create"

    def render(
        self,
        camera: Camera,
        voxel_object: VoxelObject,
        prev_model: Mat4,
        prev_viewproj: Mat4,
        frame_counter: int,
    ) -> None:
        ctx = voxel_object.voxel_texture.ctx
        self.program["m_proj"].write(camera.projection.matrix)
        self.program["m_model"].write(voxel_object.transform)
        self.program["m_model_inverse"].write(glm.inverse(voxel_object.transform))
        self.program["m_prev_model"].write(prev_model)
        self.program["m_prev_viewproj"].write(prev_viewproj)
        self.program["m_camera"].write(camera.matrix)
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.program["uInvView"].write(glm.inverse(camera.matrix))
        self.program["u_voxel_data"].value = 0
        self.program["u_palette_data"].value = 1
        self.program["frame_counter"].value = frame_counter

        voxel_object.voxel_texture.use(location=0)
        voxel_object.palette_texture.use(location=1)
        voxel_object.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


class VoxelLighting:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
        self.irradiance_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.specular_texture = window.ctx.texture(size=size, components=3, dtype="f2")

        self.ambient_lighting = VoxelAmbientLighting(window, self.irradiance_texture)
        self.direct_lighting = VoxelDirectLighting(window, self.irradiance_texture)

        self.lighting_clearer = window.ctx.framebuffer(
            color_attachments=[
                self.specular_texture,
                self.irradiance_texture,
            ]
        )
        self.lighting_clearer.label = "framebuffer_voxel_lighting_clearer"

    def render(  # noqa: PLR0913
        self,
        camera: Camera,
        gbuffer: GBuffer,
        voxel_texture: Texture3D,
        lights: Sequence[Light],
        suns: Sequence[Sun],
        frame_counter: int,
    ) -> None:
        ctx = self.irradiance_texture.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        self.lighting_clearer.clear(red=0, green=0, blue=0)

        self.ambient_lighting.render(camera, gbuffer, voxel_texture, frame_counter)

        ctx.enable_only(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD  # type:ignore[assignment]
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        for sun in suns:
            self.direct_lighting.render_sun(camera, gbuffer, voxel_texture, sun, frame_counter)
        for light in lights:
            self.direct_lighting.render_light(camera, gbuffer, voxel_texture, light, frame_counter)
        ctx.disable(moderngl.BLEND)


class VoxelAmbientLighting:
    def __init__(self, window: WindowConfig, irradiance_texture: Texture) -> None:
        self.framebuffer = window.ctx.framebuffer(color_attachments=[irradiance_texture])
        self.framebuffer.label = "framebuffer_voxel_ambient_lighting"

        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.voxel_ambient_lighting = window.load_program("programs/voxel_ambient_lighting.glsl", defines=GLOBAL_DEFINE)
        self.voxel_ambient_lighting.label = "prog_voxel_ambient_lighting"

        self.stbnormals = window.load_texture_array("assets/stbn_cosine_normals.png", layers=64)
        self.stbnormals.label = "texarr_stbn_cosine_normals"
        self.stbnormals.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def render(self, camera: Camera, gbuffer: GBuffer, voxel_texture: Texture3D, frame_counter: int) -> None:
        self.framebuffer.use()

        self.voxel_ambient_lighting["frame_counter"].value = frame_counter
        self.voxel_ambient_lighting["uProjection"].write(camera.projection.matrix)
        self.voxel_ambient_lighting["uView"].write(camera.matrix)
        self.voxel_ambient_lighting["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.voxel_ambient_lighting["uInvView"].write(glm.inverse(camera.matrix))
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        voxel_texture.use(location=3)
        self.stbnormals.use(location=4)

        self.quad_fs.render(self.voxel_ambient_lighting)


class VoxelDirectLighting:
    def __init__(self, window: WindowConfig, irradiance_texture: Texture) -> None:
        self.framebuffer = window.ctx.framebuffer(color_attachments=[irradiance_texture])
        self.framebuffer.label = "framebuffer_voxel_direct_lighting"

        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.voxel_direct_light = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"IS_SUN": 0},
        )
        self.voxel_direct_light.label = "prog_voxel_direct_light"
        self.voxel_direct_sun = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"IS_SUN": 1},
        )
        self.voxel_direct_sun.label = "prog_voxel_direct_sun"

        self.random_vec2 = window.load_texture_array("assets/stbn_vec2.png", layers=64)
        self.random_vec2.label = "texarr_stbn_vec2"
        self.random_vec2.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _setup_uniforms(self, prog: Program, camera: Camera, frame_counter: int) -> None:
        # TODO(david): This could be a context managers job, setup only once per frame, not per object
        prog["frame_counter"].value = frame_counter
        # prog["uProjection"].write(camera.projection.matrix)
        # prog["uView"].write(camera.matrix)
        prog["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        prog["uInvView"].write(glm.inverse(camera.matrix))

    def render_light(
        self,
        camera: Camera,
        gbuffer: GBuffer,
        voxel_texture: Texture3D,
        light: Light,
        frame_counter: int,
    ) -> None:
        self.framebuffer.use()
        self._setup_uniforms(self.voxel_direct_light, camera, frame_counter)

        self.voxel_direct_light["lightPos"].write(light.translation)
        self.voxel_direct_light["lightRadius"] = light.radius
        self.voxel_direct_light["lightColor"].write(light.color)
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        voxel_texture.use(location=3)
        self.random_vec2.use(location=4)

        self.quad_fs.render(self.voxel_direct_light)

    def render_sun(
        self,
        camera: Camera,
        gbuffer: GBuffer,
        voxel_texture: Texture3D,
        sun: Sun,
        frame_counter: int,
    ) -> None:
        self.framebuffer.use()
        self._setup_uniforms(self.voxel_direct_sun, camera, frame_counter)

        self.voxel_direct_sun["sunDirection"].write(sun.direction)
        self.voxel_direct_sun["lightColor"].write(sun.color)
        self.voxel_direct_sun["lightRadius"] = sun.radius
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        voxel_texture.use(location=3)
        self.random_vec2.use(location=4)

        self.quad_fs.render(self.voxel_direct_sun)
