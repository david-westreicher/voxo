from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import moderngl
import moderngl_window
from moderngl import ComputeShader, Context, Framebuffer, Program, Texture, Texture3D
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

MODEL_PATH = Path("./resources/models/truck.txt")
DWARF = Path("./resources/models/dwarf.txt")
SCREEN_DIMENSIONS = (1920, 1080)
GLOBAL_OCCLUDER_DIMENSIONS = (256, 256, 256)
CENTER = glm.vec3(GLOBAL_OCCLUDER_DIMENSIONS) * 0.5
CENTER_GROUND = glm.vec3(GLOBAL_OCCLUDER_DIMENSIONS) * 0.5
CENTER_GROUND.y = 0
ASPECT_RATIO = SCREEN_DIMENSIONS[0] / SCREEN_DIMENSIONS[1]
GL_RGB10_A2 = 0x8059
GL_DEPTH_COMPONENT32F = 0x8CAC

GLOBAL_DEFINE = {
    "SCREEN_DIMENSIONS": f"vec2({SCREEN_DIMENSIONS[0]}, {SCREEN_DIMENSIONS[1]})",
}


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

        self.debug_shader: Program = window.load_program(
            "programs/debug_occluder.glsl", defines=defines | GLOBAL_DEFINE
        )
        self.debug_shader["occluder_texture"].value = 0  # type:ignore[union-attr]
        self.debug_shader.label = "prog_debug_occluder"
        self.debug_quad = geometry.quad_fs(normals=False, uvs=True)

        self.occluder_texture = window.ctx.texture3d(size=dimensions, data=None, components=1, alignment=1, dtype="u1")
        self.occluder_texture.label = "tex3d_global_occluder"

        self.occluder_volume = Object(geometry=geometry.cube(size=dimensions))
        self.occluder_volume.translation = glm.vec3(dimensions) * 0.5

    def blit_object(self, voxel_object: VoxelObject) -> None:
        self.blitter["obj_dimensions"].write(glm.ivec3(voxel_object.model.opengl_dimensions))  # type:ignore[union-attr]
        self.blitter["obj_transform_inv"].write(glm.inverse(voxel_object.transform))  # type:ignore[union-attr]
        voxel_object.voxel_texture.use(location=0)
        self.occluder_texture.bind_to_image(1, read=False, write=True, level=0)
        self.blitter.run(
            GLOBAL_OCCLUDER_DIMENSIONS[0] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[1] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[2] // 8,
        )

    def clear(self) -> None:
        # TODO(david): We should use glClearTexImage/glClearTexSubImage but it's not supported in moderngl
        self.occluder_texture.bind_to_image(0, read=False, write=True, level=0)
        self.clearer.run(
            GLOBAL_OCCLUDER_DIMENSIONS[0] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[1] // 8,
            GLOBAL_OCCLUDER_DIMENSIONS[2] // 8,
        )

    def render_debug(self, camera: Camera) -> None:
        self.occluder_texture.use(location=0)
        self.debug_shader["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.debug_shader["uInvProjection"].write(glm.inverse(camera.projection.matrix))  # type:ignore[union-attr]
        self.debug_shader["uInvView"].write(glm.inverse(camera.matrix))  # type:ignore[union-attr]
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
        self.program["m_proj"].write(camera.projection.matrix)  # type:ignore[union-attr]
        self.program["m_model"].write(voxel_object.transform)  # type:ignore[union-attr]
        self.program["m_model_inverse"].write(glm.inverse(voxel_object.transform))  # type:ignore[union-attr]
        self.program["m_prev_model"].write(prev_model)  # type:ignore[union-attr]
        self.program["m_prev_viewproj"].write(prev_viewproj)  # type:ignore[union-attr]
        self.program["m_camera"].write(camera.matrix)  # type:ignore[union-attr]
        self.program["uInvProjection"].write(glm.inverse(camera.projection.matrix))  # type:ignore[union-attr]
        self.program["uInvView"].write(glm.inverse(camera.matrix))  # type:ignore[union-attr]
        self.program["uCameraPos"].write(camera.position)  # type:ignore[union-attr]
        self.program["u_voxel_data"].value = 0  # type:ignore[union-attr]
        self.program["u_palette_data"].value = 1  # type:ignore[union-attr]
        self.program["frame_counter"].value = frame_counter  # type:ignore[union-attr]

        voxel_object.voxel_texture.use(location=0)
        voxel_object.palette_texture.use(location=1)
        voxel_object.geometry.render(self.program)
        ctx.enable(moderngl.DEPTH_TEST)


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
        # clear depth buffer
        self.framebuffer.color_mask = [(False,) * 4] * len(self.framebuffer.color_attachments)
        self.framebuffer.clear(depth=1.0)

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
        self.quad_fs.render(self.gbuffer_lighting)

        self.pingpong = 1 - self.pingpong


class GBufferDebug:
    def __init__(
        self,
        window: moderngl_window.WindowConfig,  # type: ignore[name-defined]
        gbuffer_lighting_texture: moderngl.Texture,
    ) -> None:
        self.quad_fs = geometry.quad_fs(normals=False, uvs=True)
        self.gbuffer_debug = window.load_program("programs/gbuffer_debug.glsl")
        self.gbuffer_debug.label = "prog_gbuffer_debug"
        self.gbuffer_debug["u_albedo"].value = 0
        self.gbuffer_debug["u_normal"].value = 1
        self.gbuffer_debug["u_depth"].value = 2
        self.gbuffer_debug["u_motion_vectors"].value = 3
        self.gbuffer_debug["u_lighting"].value = 4
        self.gbuffer_lighting = gbuffer_lighting_texture

    def render(self, gbuffer: GBuffer, *, debug: bool) -> None:
        gbuffer.albedo_texture.use(location=0)
        gbuffer.smooth_normal_texture.use(location=1)
        gbuffer.linear_depth.use(location=2)
        gbuffer.motion_vectors.use(location=3)
        self.gbuffer_lighting.use(location=4)
        self.gbuffer_debug["full"].value = not debug
        self.quad_fs.render(self.gbuffer_debug)


class WireFrameRenderer:
    def __init__(self, window: moderngl_window.WindowConfig) -> None:  # type: ignore[name-defined]
        self.prog = window.load_program("programs/cube_simple.glsl")
        self.prog.label = "prof_cube_simple"
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


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.truck = VoxelObject(model=parse_model(Path("./resources/models/truck.txt")))
        self.dwarf = VoxelObject(model=parse_model(Path("./resources/models/dwarf.txt")))
        self.plane = VoxelObject(model=parse_model(Path("./resources/models/plane.txt")))
        self.light = Object(geometry.sphere(2))

        self.last_frame_transforms = [obj.transform for obj in self.voxel_objects]
        for voxel_object in self.voxel_objects:
            voxel_object.upload_to_gpu(ctx)

    @cached_property
    def voxel_objects(self) -> Sequence[VoxelObject]:
        return [self.truck, self.dwarf, self.plane]

    def update(self, time: float) -> None:
        self.truck.translation = glm.vec3(0, self.truck.model.dimensions[2] * 0.5 + 1, 0) + CENTER_GROUND
        self.dwarf.translation = glm.vec3(0, 58.5, 0) + CENTER_GROUND
        self.light.translation = glm.rotateY(glm.vec3(80, 0, 0), time) + CENTER
        self.plane.translation = CENTER_GROUND
        self.truck.rotate(0.001, glm.vec3(0, 1, 0))
        self.dwarf.rotate(-0.001, glm.vec3(0, 1, 0))

    def update_lastframe_transforms(self) -> None:
        for i, voxel_object in enumerate(self.voxel_objects):
            self.last_frame_transforms[i] = voxel_object.transform


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
            # self.wireframe_box.render(self.camera, self.scene.voxel_objects)
            # self.wireframe_box.render(self.camera, [self.scene.light, self.global_occluder.occluder_volume])
        self.gbuffer.swap()
