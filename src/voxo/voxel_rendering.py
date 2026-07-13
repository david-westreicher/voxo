from dataclasses import dataclass, field
from typing import cast

import moderngl
from moderngl import ComputeShader, Context, Framebuffer, Program, Texture, Texture3D
from moderngl_window import geometry
from moderngl_window.context.base import WindowConfig
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene import Camera
from pyglm import glm
from pyglm.glm import mat4x4 as Mat4  # noqa: N812
from pyglm.glm import vec3 as Vec3  # noqa: N812

from .constants import GLOBAL_DEFINE, GLOBAL_OCCLUDER_DIMENSIONS
from .model import Model
from .rendering import GBuffer
from .utils import Object


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
            mip_maps=True,
        )
        self._voxel_texture.label = f"tex3d_model_{self.model.name}"
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


class GlobalOccluder:
    def __init__(self, window: WindowConfig, dimensions: tuple[int, int, int]) -> None:
        self.dimensions = dimensions
        defines = {"GLOBAL_DIMENSIONS": f"ivec3({','.join(f'{d}.0' for d in dimensions)})"}
        self.blitter: ComputeShader = window.load_compute_shader("programs/blitter.glsl", defines=defines)
        self.blitter["voxel_texture"].value = 0
        self.blitter.label = "prog_blitter"

        self.clearer: ComputeShader = window.load_compute_shader("programs/clearer.glsl", defines=defines)
        self.clearer.label = "prog_clearer"

        self.mipmapper: ComputeShader = window.load_compute_shader("programs/occluder_mipmapper.glsl", defines=defines)
        self.mipmapper.label = "prog_occluder_mipmapper"

        self.debug_shader: Program = window.load_program(
            "programs/debug_occluder.glsl", defines=defines | GLOBAL_DEFINE
        )
        self.debug_shader["occluder_texture"].value = 0
        self.debug_shader.label = "prog_debug_occluder"
        self.debug_quad = geometry.quad_fs(normals=False, uvs=True)

        self.occluder_texture = window.ctx.texture3d(
            size=dimensions,
            data=None,
            components=1,
            alignment=1,
            dtype="u1",
            mip_maps=True,
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
        self.debug_shader["size"].write(glm.ivec3(GLOBAL_OCCLUDER_DIMENSIONS))
        self.debug_shader["uCameraPos"].write(camera.position)
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
        self.program["uCameraPos"].write(camera.position)
        self.program["u_voxel_data"].value = 0
        self.program["u_palette_data"].value = 1
        self.program["frame_counter"].value = frame_counter

        voxel_object.voxel_texture.use(location=0)
        voxel_object.palette_texture.use(location=1)
        voxel_object.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


class VoxelLighting:
    def __init__(
        self,
        window: WindowConfig,
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
