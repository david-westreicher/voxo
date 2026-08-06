import struct
from collections.abc import Sequence
from functools import cached_property
from itertools import groupby
from typing import cast

import moderngl
import moderngl_window
from moderngl import ComputeShader, Program, Texture
from moderngl_window import geometry
from moderngl_window.context.base import WindowConfig
from moderngl_window.scene import Camera
from pyglm import glm

from .constants import (
    GLOBAL_DEFINE,
    LIGHT_TYPE_NUM_AREA,
    LIGHT_TYPE_NUM_CONE,
    LIGHT_TYPE_NUM_SPHERE,
    LIGHT_TYPE_NUM_SUN,
    MAX_VOXEL_OBJECTS,
    USE_VOXEL_OBJECT_INSTANCING,
    VOXEL_OBJECT_COUNT_PER_BATCH,
)
from .objects import AreaLight, ConeLight, Light, Object, SphereLight, Sun, VoxelObject
from .rendering import Denoiser, GBuffer
from .utils import chunk_iters


class GlobalOccluder:
    def __init__(self, window: WindowConfig, dimensions: tuple[int, int, int], *, center: bool = False) -> None:
        self.dimensions = dimensions
        self.blitter: ComputeShader = window.load_compute_shader("programs/blitter.glsl")
        self.blitter.label = "prog_blitter"

        self.clearer: ComputeShader = window.load_compute_shader("programs/clearer.glsl")
        self.clearer.label = "prog_clearer"

        self.mipmapper: ComputeShader = window.load_compute_shader("programs/occluder_mipmapper.glsl")
        self.mipmapper.label = "prog_occluder_mipmapper"

        self.debug_shader: Program = window.load_program("programs/debug_occluder.glsl", defines=GLOBAL_DEFINE)
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
        self.occluder_texture.repeat_x = False
        self.occluder_texture.repeat_y = False
        self.occluder_texture.repeat_z = False
        self.occluder_texture.label = "tex3d_global_occluder"

        self.occluder_volume = Object(
            geometry=geometry.cube(
                size=dimensions,
                center=(glm.vec3(dimensions) * 0.5).to_tuple(),
            )
        )
        if center:
            self.occluder_volume.translation = -cast("glm.vec3", glm.ceil(glm.vec3(dimensions) * 0.5))
            self.occluder_volume.translation.y = 0

    def blit_object(self, voxel_object: VoxelObject) -> None:
        min_aabb_vec, max_aabb_vec = voxel_object.aabb
        min_aabb_vec -= self.occluder_volume.translation
        max_aabb_vec -= self.occluder_volume.translation
        min_aabb = glm.clamp(glm.ivec3(glm.floor(min_aabb_vec)), glm.ivec3(0), glm.ivec3(self.dimensions)).to_tuple()
        max_aabb = glm.clamp(
            glm.ivec3(glm.ceil(max_aabb_vec)), min_aabb + glm.ivec3(1), glm.ivec3(self.dimensions)
        ).to_tuple()
        size = (max_aabb[0] - min_aabb[0], max_aabb[1] - min_aabb[1], max_aabb[2] - min_aabb[2])
        if any(dim == 0 for dim in size):
            return
        for min_coord, max_coord, dim in zip(min_aabb, max_aabb, self.dimensions, strict=True):
            assert 0 <= min_coord < max_coord <= dim

        self.blitter["obj_transform_inv"].write(glm.inverse(voxel_object.transform))
        self.blitter["min_cell"] = min_aabb
        self.blitter["max_cell"] = max_aabb
        self.blitter["occluder_translation"].write(glm.ivec3(self.occluder_volume.translation))
        self.blitter["material_row"] = voxel_object.model.material_row
        voxel_object.voxel_texture.use(location=0)
        self.occluder_texture.bind_to_image(1, read=False, write=True, level=0)
        voxel_object.texture_information.material_texture.use(location=2)
        self.blitter.run(
            (size[0] + 7) // 8,
            (size[1] + 7) // 8,
            (size[2] + 7) // 8,
        )

    def clear_region(self, min_cell: tuple[int, int, int], max_cell: tuple[int, int, int]) -> None:
        for min_coord, max_coord, dim in zip(min_cell, max_cell, self.dimensions, strict=True):
            assert min_coord < max_coord <= dim
        size = (max_cell[0] - min_cell[0], max_cell[1] - min_cell[1], max_cell[2] - min_cell[2])
        self.occluder_texture.bind_to_image(0, read=False, write=True, level=0)
        self.clearer["min_cell"] = min_cell
        self.clearer["max_cell"] = max_cell
        self.clearer.run(
            (size[0] + 7) // 8,
            (size[1] + 7) // 8,
            (size[2] + 7) // 8,
        )

    def clear(self) -> None:
        self.clear_region((0, 0, 0), self.dimensions)

    def update_mipmaps(self) -> None:
        destination_dimensions = glm.ivec3(self.dimensions) // 2
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
        self.debug_shader["occluder_translation"].write(glm.ivec3(self.occluder_volume.translation))
        self.debug_shader["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.debug_shader["uInvView"].write(glm.inverse(camera.matrix))
        self.debug_quad.render(self.debug_shader)


class VoxelRenderer:
    def __init__(self, window: WindowConfig) -> None:
        self.program: Program = window.load_program("programs/gbuffer_create.glsl", defines=GLOBAL_DEFINE)
        self.program.label = "prog_gbuffer_create"
        self.object_transform_buffer = window.ctx.buffer(reserve=(64 * 3 + 16) * MAX_VOXEL_OBJECTS, dynamic=True)
        self.object_voxel_texture_handle_buffer = window.ctx.buffer(reserve=(8 * 3) * MAX_VOXEL_OBJECTS, dynamic=True)
        self.cube = geometry.cube((1, 1, 1), (0.5, 0.5, 0.5))

    def render_objects(
        self,
        voxel_objects: list[VoxelObject],
        camera: Camera,
        prev_viewproj: glm.mat4x4,
        prev_linear_depth_texture: Texture,
        frame_counter: int,
    ) -> None:
        if len(voxel_objects) == 0:
            return
        assert all(obj.texture_information == voxel_objects[0].texture_information for obj in voxel_objects)
        texture_information = voxel_objects[0].texture_information
        voxel_objects.sort(key=lambda obj: glm.distance2(camera.position, obj.center))

        ctx = self.program.ctx
        self.program["m_proj"].write(camera.projection.matrix)
        self.program["m_prev_viewproj"].write(prev_viewproj)
        self.program["m_camera"].write(camera.matrix)
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.program["uInvView"].write(glm.inverse(camera.matrix))
        self.program["frame_counter"].value = frame_counter
        texture_information.palette_texture.use(location=1)
        texture_information.material_texture.use(location=2)
        prev_linear_depth_texture.use(location=3)

        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.cull_face = "front"

        batched_voxel_objects = chunk_iters(voxel_objects, VOXEL_OBJECT_COUNT_PER_BATCH)
        for b_voxel_objects in batched_voxel_objects:
            transform_buffer = []
            for voxel_object in b_voxel_objects:
                transform_buffer.append(voxel_object.transform.to_bytes())
                transform_buffer.append(glm.inverse(voxel_object.transform).to_bytes())
                transform_buffer.append(voxel_object.last_frame_transform.to_bytes())
                transform_buffer.append(glm.vec4(*voxel_object.model.opengl_dimensions, 1).to_bytes())
            self.object_transform_buffer.write(b"".join(transform_buffer))
            self.object_transform_buffer.bind_to_storage_buffer(binding=0)
            if USE_VOXEL_OBJECT_INSTANCING:
                texture_handle_buffer = []
                for voxel_object in b_voxel_objects:
                    texture_handle_buffer.append(struct.pack("<Q", voxel_object.voxel_texture_handle))
                    texture_handle_buffer.append(struct.pack("<I", voxel_object.model.palette_row))
                    texture_handle_buffer.append(struct.pack("<I", voxel_object.model.material_row))
                self.object_voxel_texture_handle_buffer.write(b"".join(texture_handle_buffer))
                self.object_voxel_texture_handle_buffer.bind_to_storage_buffer(binding=1)
                self.cube.render(self.program, instances=len(b_voxel_objects))
            else:
                for i, voxel_object in enumerate(b_voxel_objects):
                    assert voxel_object.visible
                    voxel_object.voxel_texture.use(location=0)
                    self.program["u_instanceID"] = i
                    self.program["u_palette_row"] = voxel_object.model.palette_row
                    self.program["u_material_row"] = voxel_object.model.material_row
                    self.cube.render(self.program)
        ctx.cull_face = "back"
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.program]


class VoxelLighting:
    def __init__(self, window: WindowConfig, size: tuple[int, int]) -> None:
        self.irradiance_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.irradiance_texture.label = "tex_irradiance_texture"
        self.specular_texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.specular_texture.label = "tex_specular_texture"
        self.reflectivity_texture = window.ctx.texture(size=size, components=1, dtype="f2")
        self.reflectivity_texture.label = "tex_reflectivity_texture"

        self.ambient_lighting = VoxelAmbientLighting(window, self.irradiance_texture)
        self.direct_lighting = VoxelDirectLighting(window, self.irradiance_texture)
        self.specular_lighting = VoxelSpecularLighting(window, self.specular_texture, self.reflectivity_texture)

        self.irradiance_denoiser_1 = Denoiser(window, size, "irradiance_denoiser_1")
        self.irradiance_denoiser_2 = Denoiser(window, size, "irradiance_denoiser_2")
        self.specular_denoiser = Denoiser(window, size, "specular_denoiser")
        self.compositor = LightCompositor(window, size)

        self.lighting_clearer = window.ctx.framebuffer(
            color_attachments=[
                self.specular_texture,
                self.irradiance_texture,
                self.reflectivity_texture,
            ]
        )
        self.lighting_clearer.label = "framebuffer_voxel_lighting_clearer"

    def clear(self) -> None:
        ctx = self.irradiance_texture.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        self.lighting_clearer.clear(red=0, green=0, blue=0)

    def render_ambient(
        self,
        camera: Camera,
        current_gbuffer: GBuffer,
        occluder: GlobalOccluder,
        frame_counter: int,
    ) -> None:
        self.ambient_lighting.render(camera, current_gbuffer, occluder, frame_counter)

    def render_direct(  # noqa: PLR0913
        self,
        camera: Camera,
        current_gbuffer: GBuffer,
        occluder: GlobalOccluder,
        lights: Sequence[Light],
        suns: Sequence[Sun],
        frame_counter: int,
    ) -> None:
        ctx = self.irradiance_texture.ctx
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD  # type:ignore[assignment]
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        for sun in suns:
            if not sun.visible:
                continue
            self.direct_lighting.render_sun(camera, current_gbuffer, occluder, sun, frame_counter)
        self.direct_lighting.render_lights(camera, current_gbuffer, occluder, lights, frame_counter)
        ctx.disable(moderngl.BLEND)

    def denoise_direct(
        self,
        camera: Camera,
        current_gbuffer: GBuffer,
        last_gbuffer: GBuffer,
        frame_counter: int,
        *,
        camera_moved: bool,
    ) -> None:
        self.irradiance_denoiser_1.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=self.irradiance_texture,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter,
            last_texture=self.irradiance_denoiser_2.clean_texture,
        )
        self.irradiance_denoiser_2.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=self.irradiance_denoiser_1.clean_texture,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter + 1,
        )
        self.compositor.composite_diffuse(current_gbuffer, self.irradiance_denoiser_2.clean_texture)

    def render_specular(  # noqa: PLR0913
        self,
        camera: Camera,
        current_gbuffer: GBuffer,
        last_gbuffer: GBuffer,
        occluder: GlobalOccluder,
        frame_counter: int,
        *,
        camera_moved: bool,
    ) -> None:
        self.specular_lighting.render(
            camera,
            current_gbuffer,
            occluder,
            self.compositor.output_texture,
            frame_counter,
        )
        self.specular_denoiser.render(
            camera=camera,
            camera_moved=camera_moved,
            current_texture=self.specular_texture,
            motion_vectors=current_gbuffer.motion_vectors,
            current_depth=current_gbuffer.linear_depth,
            last_depth=last_gbuffer.linear_depth,
            current_normals=current_gbuffer.normal_texture,
            frame_counter=frame_counter,
        )
        self.compositor.composite_specular(
            current_gbuffer,
            self.specular_denoiser.clean_texture,
            self.reflectivity_texture,
        )

    @property
    def final_light_texture(self) -> Texture:
        return self.compositor.output_texture

    @cached_property
    def textures(self) -> list[Texture]:
        return [
            self.irradiance_texture,
            self.specular_texture,
            self.reflectivity_texture,
            *self.compositor.textures,
            *self.irradiance_denoiser_1.textures,
            *self.irradiance_denoiser_2.textures,
            *self.specular_denoiser.textures,
        ]

    @cached_property
    def shaders(self) -> list[Program]:
        return [
            *self.ambient_lighting.shaders,
            *self.direct_lighting.shaders,
            *self.specular_lighting.shaders,
            *self.compositor.shaders,
            *self.irradiance_denoiser_1.shaders,
            *self.irradiance_denoiser_2.shaders,
            *self.specular_denoiser.shaders,
        ]


class VoxelAmbientLighting:
    def __init__(self, window: WindowConfig, irradiance_texture: Texture) -> None:
        self.framebuffer = window.ctx.framebuffer(color_attachments=[irradiance_texture])
        self.framebuffer.label = "framebuffer_voxel_ambient_lighting"

        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.voxel_ambient_lighting = window.load_program("programs/voxel_ambient_lighting.glsl", defines=GLOBAL_DEFINE)
        self.voxel_ambient_lighting.label = "prog_voxel_ambient_lighting"
        self.voxel_ambient_lighting["max_occ_samples"] = 2
        self.voxel_ambient_lighting["ambient_strength"] = 1.0

        self.stbnormals = window.load_texture_array("assets/stbn_cosine_normals.png", layers=64)
        self.stbnormals.label = "texarr_stbn_cosine_normals"
        self.stbnormals.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.stbn_vec3 = window.load_texture_array("assets/stbn_vec3.png", layers=64)
        self.stbn_vec3.label = "texarr_stbn_vec3"
        self.stbn_vec3.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def render(self, camera: Camera, gbuffer: GBuffer, occluder: GlobalOccluder, frame_counter: int) -> None:
        self.framebuffer.use()

        self.voxel_ambient_lighting["occluder_translation"].write(glm.ivec3(occluder.occluder_volume.translation))
        self.voxel_ambient_lighting["frame_counter"].value = frame_counter
        self.voxel_ambient_lighting["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.voxel_ambient_lighting["uInvView"].write(glm.inverse(camera.matrix))
        gbuffer.normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        occluder.occluder_texture.use(location=3)
        self.stbnormals.use(location=4)
        self.stbn_vec3.use(location=5)

        self.quad_fs.render(self.voxel_ambient_lighting)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.voxel_ambient_lighting]


class VoxelDirectLighting:
    def __init__(self, window: WindowConfig, irradiance_texture: Texture) -> None:
        self.framebuffer = window.ctx.framebuffer(color_attachments=[irradiance_texture])
        self.framebuffer.label = "framebuffer_voxel_direct_lighting"

        self.voxel_direct_sun = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"LIGHT_TYPE": f"{LIGHT_TYPE_NUM_SUN}"},
        )
        self.voxel_direct_sun.label = "prog_voxel_direct_sun"

        self.voxel_direct_sphere = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"LIGHT_TYPE": f"{LIGHT_TYPE_NUM_SPHERE}"},
        )
        self.voxel_direct_sphere.label = "prog_voxel_direct_sphere"

        self.voxel_direct_cone = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"LIGHT_TYPE": f"{LIGHT_TYPE_NUM_CONE}"},
        )
        self.voxel_direct_cone.label = "prog_voxel_direct_cone"

        self.voxel_direct_area = window.load_program(
            "programs/voxel_direct_lighting.glsl",
            defines=GLOBAL_DEFINE | {"LIGHT_TYPE": f"{LIGHT_TYPE_NUM_AREA}"},
        )
        self.voxel_direct_area.label = "prog_voxel_direct_area"

        self.random_vec2 = window.load_texture_array("assets/stbn_vec2.png", layers=64)
        self.random_vec2.label = "texarr_stbn_vec2"
        self.random_vec2.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)

    def _setup_uniforms(
        self,
        prog: Program,
        camera: Camera,
        occluder: GlobalOccluder,
        frame_counter: int | None = None,
    ) -> None:
        # TODO(david): This could be a context managers job, setup only once per frame, not per object
        if frame_counter is not None:
            prog["frame_counter"].value = frame_counter
        prog["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        prog["uInvView"].write(glm.inverse(camera.matrix))
        prog["occluder_translation"].write(glm.ivec3(occluder.occluder_volume.translation))
        if "m_proj" in prog:
            prog["m_proj"].write(camera.projection.matrix)
        if "m_camera" in prog:
            prog["m_camera"].write(camera.matrix)

    def render_lights(
        self,
        camera: Camera,
        gbuffer: GBuffer,
        occluder: GlobalOccluder,
        lights: Sequence[Light],
        frame_counter: int,
    ) -> None:
        visible_lights = [light for light in lights if light.visible]
        visible_lights.sort(key=lambda x: str(type(x)))
        light_by_type_mapping = {k: list(v) for k, v in groupby(visible_lights, type)}
        sphere_lights: list[SphereLight] = cast("list[SphereLight]", light_by_type_mapping.get(SphereLight, []))
        cone_lights: list[ConeLight] = cast("list[ConeLight]", light_by_type_mapping.get(ConeLight, []))
        area_lights: list[AreaLight] = cast("list[AreaLight]", light_by_type_mapping.get(AreaLight, []))

        self.framebuffer.use()
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        occluder.occluder_texture.use(location=3)
        self.random_vec2.use(location=4)

        ctx = self.random_vec2.ctx
        prev_cull_face = ctx.cull_face
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "front"
        if sphere_lights:
            self.render_sphere_lights(camera, occluder, sphere_lights, frame_counter)
        if cone_lights:
            self.render_cone_lights(camera, occluder, cone_lights)
        if area_lights:
            self.render_area_lights(camera, occluder, area_lights, frame_counter)
        ctx.disable(moderngl.CULL_FACE)
        ctx.cull_face = prev_cull_face

    def render_sphere_lights(
        self, camera: Camera, occluder: GlobalOccluder, lights: Sequence[SphereLight], frame_counter: int
    ) -> None:
        self._setup_uniforms(self.voxel_direct_sphere, camera, occluder, frame_counter)

        for light in lights:
            self.voxel_direct_sphere["unshadowed"] = light.unshadowed
            self.voxel_direct_sphere["lightPos"].write(light.translation)
            self.voxel_direct_sphere["lightColor"].write(light.color * light.intensity)
            self.voxel_direct_sphere["lightRadius"] = light.light_size
            self.voxel_direct_sphere["reach"] = light.reach
            self.voxel_direct_sphere["m_model"].write(light.proxy_transform)
            light.proxy_geometry.render(self.voxel_direct_sphere)

    def render_cone_lights(self, camera: Camera, occluder: GlobalOccluder, lights: Sequence[ConeLight]) -> None:
        self._setup_uniforms(self.voxel_direct_cone, camera, occluder, frame_counter=None)

        for light in lights:
            self.voxel_direct_cone["unshadowed"] = light.unshadowed
            self.voxel_direct_cone["lightColor"].write(light.color * light.intensity)
            self.voxel_direct_cone["lightPos"].write(light.translation)
            self.voxel_direct_cone["lightDirection"].write(light.direction)
            self.voxel_direct_cone["penumbraCos"] = glm.cos(glm.radians(light.penumbra))
            self.voxel_direct_cone["reach"] = light.reach
            self.voxel_direct_cone["m_model"].write(light.proxy_transform)
            light.proxy_geometry.render(self.voxel_direct_cone)

    def render_area_lights(
        self, camera: Camera, occluder: GlobalOccluder, lights: Sequence[AreaLight], frame_counter: int
    ) -> None:
        self._setup_uniforms(self.voxel_direct_area, camera, occluder, frame_counter)

        for light in lights:
            self.voxel_direct_area["unshadowed"] = light.unshadowed
            self.voxel_direct_area["lightColor"].write(light.color * light.intensity)
            self.voxel_direct_area["light_matrix"].write(light.area_light_matrix)
            self.voxel_direct_area["reach"] = light.reach
            self.voxel_direct_area["m_model"].write(light.proxy_transform)
            light.proxy_geometry.render(self.voxel_direct_area)

    def render_sun(
        self,
        camera: Camera,
        gbuffer: GBuffer,
        occluder: GlobalOccluder,
        sun: Sun,
        frame_counter: int,
    ) -> None:
        self.framebuffer.use()
        self._setup_uniforms(self.voxel_direct_sun, camera, occluder, frame_counter)

        self.voxel_direct_sun["occluder_translation"].write(glm.ivec3(occluder.occluder_volume.translation))
        self.voxel_direct_sun["sunDirection"].write(sun.direction)
        self.voxel_direct_sun["lightColor"].write(sun.color)
        self.voxel_direct_sun["lightRadius"] = sun.radius
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        occluder.occluder_texture.use(location=3)
        self.random_vec2.use(location=4)

        self.quad_fs.render(self.voxel_direct_sun)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.voxel_direct_sphere, self.voxel_direct_sun]


class VoxelSpecularLighting:
    def __init__(self, window: WindowConfig, specular_texture: Texture, reflictivity_texture: Texture) -> None:
        self.framebuffer = window.ctx.framebuffer(color_attachments=[specular_texture, reflictivity_texture])
        self.framebuffer.label = "framebuffer_voxel_specular_lighting"

        self.voxel_specular_lighting = window.load_program(
            "programs/voxel_specular_lighting.glsl", defines=GLOBAL_DEFINE
        )
        self.voxel_specular_lighting.label = "prog_voxel_specular_lighting"

        self.stbnormals = window.load_texture_array("assets/stbn_unitvec3.png", layers=64)
        self.stbnormals.label = "texarr_stbn_unitvec3"
        self.stbnormals.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)

    def render(
        self,
        camera: Camera,
        gbuffer: GBuffer,
        occluder: GlobalOccluder,
        color_texture: Texture,
        frame_counter: int,
    ) -> None:
        self.framebuffer.use()

        self.voxel_specular_lighting["occluder_translation"].write(glm.ivec3(occluder.occluder_volume.translation))
        self.voxel_specular_lighting["uInvProjection"].write(glm.inverse(camera.projection.matrix))
        self.voxel_specular_lighting["uInvView"].write(glm.inverse(camera.matrix))
        self.voxel_specular_lighting["u_projection_view"].write(camera.projection.matrix @ camera.matrix)
        self.voxel_specular_lighting["frame_counter"] = frame_counter
        gbuffer.smooth_normal_texture.use(location=0)
        gbuffer.depth_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        gbuffer.material_texture.use(location=3)
        occluder.occluder_texture.use(location=4)
        color_texture.use(location=5)
        self.stbnormals.use(location=6)

        self.quad_fs.render(self.voxel_specular_lighting)

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.voxel_specular_lighting]


class LightCompositor:
    def __init__(self, window: moderngl_window.WindowConfig, size: tuple[int, int]) -> None:  # type: ignore[name-defined]
        self.output_texture: Texture = window.ctx.texture(size=size, components=3, dtype="f2")
        self.output_texture.label = "tex2d_light_composite"
        self.output_texture.repeat_x = False
        self.output_texture.repeat_y = False
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.output_texture])
        self.framebuffer.label = "framebuffer_light_composite"

        self.comp_diffuse = window.load_program("programs/composite_diffuse.glsl", defines=GLOBAL_DEFINE)
        self.comp_diffuse.label = "prog_composite_diffuse"
        self.comp_specular = window.load_program("programs/composite_specular.glsl", defines=GLOBAL_DEFINE)
        self.comp_specular.label = "prog_composite_specular"
        self.quad = geometry.quad_fs(normals=False, uvs=True)

    def composite_diffuse(self, gbuffer: GBuffer, clean_diffuse: Texture) -> None:
        self.framebuffer.use()

        gbuffer.albedo_texture.use(location=0)
        clean_diffuse.use(location=1)
        gbuffer.material_texture.use(location=2)
        gbuffer.depth_texture.use(location=3)
        self.quad.render(self.comp_diffuse)

    def composite_specular(self, gbuffer: GBuffer, clean_specular: Texture, reflectivity: Texture) -> None:
        self.framebuffer.use()

        gbuffer.albedo_texture.use(location=0)
        self.output_texture.use(location=1)
        clean_specular.use(location=2)
        gbuffer.material_texture.use(location=3)
        reflectivity.use(location=4)
        gbuffer.depth_texture.use(location=5)
        self.quad.render(self.comp_specular)

    @cached_property
    def textures(self) -> list[Texture]:
        return [self.output_texture]

    @cached_property
    def shaders(self) -> list[Program]:
        return [self.comp_diffuse]
