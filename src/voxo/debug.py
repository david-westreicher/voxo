from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import cast

import moderngl
import numpy as np
from imgui_bundle import ImVec2, ImVec4, imgui, implot
from moderngl import Context, Program, Query, Texture, Uniform
from moderngl_window import geometry
from moderngl_window.context.base.window import WindowConfig
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer
from pyglm import glm

from voxo.objects import Light, Object, VoxelObject

from .model import parse_model
from .scene import Scene


class Profiler:
    def __init__(self, ctx: Context) -> None:
        self.query_buffer: dict[tuple[str, int], Query] = defaultdict(lambda: ctx.query(time=True))

    @contextmanager
    def query(self, name: str, frame_counter: int) -> Iterator[None]:
        query = self.query_buffer[(name, frame_counter % 10)]
        query.__enter__()  # type:ignore[no-untyped-call]
        try:
            yield
        finally:
            query.__exit__()

    def timing(self, name: str, frame_counter: int) -> int:
        key = (name, (frame_counter - 3) % 10)
        if key not in self.query_buffer:
            return 0
        query = self.query_buffer[key]
        return query.elapsed

    def all_timings(self, frame_counter: int, frame_time: float) -> dict[str, int]:
        names = [name for name, f in self.query_buffer if f == (frame_counter - 3) % 10]
        timings = {name: self.timing(name, frame_counter) for name in names}
        timings["frame time"] = int(frame_time * 1_000_000_000)
        return timings


class ScrollingBuffer:
    """Simulates a scrolling buffer for real-time plotting."""

    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self.offset = 0
        self.data = np.empty((max_size, 2), dtype=np.float32)
        self.size = 0

    def add_point(self, x: float, y: float) -> None:
        if self.size < self.max_size:
            self.data[self.size] = [x, y]
            self.size += 1
        else:
            self.data[self.offset] = [x, y]
            self.offset = (self.offset + 1) % self.max_size

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the data as contiguous 1D arrays for plotting."""
        if self.size < self.max_size:
            data = self.data[: self.size]
        else:
            # Ensure data is contiguous after roll operation
            data = np.ascontiguousarray(np.roll(self.data, -self.offset, axis=0))

        # Extract columns and create fresh contiguous arrays
        # Using .copy() ensures we get a truly independent, contiguous array
        xs = data[:, 0].astype(np.float32, order="C", copy=True)
        ys = data[:, 1].astype(np.float32, order="C", copy=True)

        return xs, ys


class TextureViewer:
    PREVIEW_SIZE = 300

    def __init__(self, textures: list[Texture], window: WindowConfig) -> None:
        self.textures = textures
        self.selected_texture = 0
        self.zoom_pixel = 5
        self.zoom_pos = glm.vec2(self.zoom_pixel, self.zoom_pixel)
        self.filter_min = 0.0
        self.filter_max = 1.0

        self.preview_texture = window.ctx.texture(size=(2000, 2000), components=3, dtype="f2")
        self.preview_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.framebuffer = window.ctx.framebuffer(color_attachments=[self.preview_texture])
        self.framebuffer.label = "framebuffer_debug_texture_viewer"
        self.program = window.load_program("programs/debug_texture_viewer_preview.glsl")
        self.program.label = "prog_debug_texture_viewer_preview"
        self.quad = geometry.quad_fs(normals=False, uvs=True)
        self.ctx = window.ctx

    def render_into_preview_texture(self, texture: Texture) -> None:
        prev_framebuffer = self.ctx.fbo
        self.framebuffer.use()
        self.program["filter_min"] = self.filter_min
        self.program["filter_max"] = self.filter_max
        texture.use(location=0)
        self.quad.render(self.program)

        prev_framebuffer.use()

    def render(self) -> None:
        imgui.set_next_window_size(imgui.ImVec2(1300, 1000), imgui.Cond_.appearing)  # type:ignore[arg-type]
        imgui.begin("Textures", p_open=True)

        imgui.begin_child("texture_list", (280, 0), child_flags=True)
        for i, tex in enumerate(self.textures):
            assert tex.label
            clicked, _ = imgui.selectable(tex.label, self.selected_texture == i)
            if clicked:
                self.selected_texture = i
        imgui.end_child()

        imgui.same_line()

        imgui.begin_group()
        selected_texture = self.textures[self.selected_texture]
        w, h = selected_texture.size
        imgui.text(f"Name   : {selected_texture.label}")
        imgui.text(f"Size   : {w} x {h}")
        imgui.separator()

        self.render_into_preview_texture(selected_texture)
        origin = imgui.get_cursor_screen_pos()
        image_size = glm.ivec2(1000, 1000 * h / w)
        col_red = imgui.get_color_u32(ImVec4(1, 0, 0, 1.0))
        imgui.image(self.preview_texture.glo, image_size.to_tuple(), uv0=ImVec2(0, 1), uv1=ImVec2(1, 0))
        if imgui.is_mouse_dragging(imgui.MouseButton_.right):  # type:ignore[arg-type]
            mouse = imgui.get_mouse_pos()
            image_pos = imgui.get_item_rect_min()
            self.zoom_pos.x = mouse.x - image_pos.x
            self.zoom_pos.y = mouse.y - image_pos.y
        image_zoom_pixel = self.zoom_pixel * image_size.x / selected_texture.size[0]
        offset = image_zoom_pixel / 2
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect(
            ImVec2(origin.x + self.zoom_pos.x - offset, origin.y + self.zoom_pos.y - offset),
            ImVec2(
                origin.x + self.zoom_pos.x + image_zoom_pixel - offset,
                origin.y + self.zoom_pos.y + image_zoom_pixel - offset,
            ),
            col_red,
            thickness=0.1,
        )
        inverted_zoom_pos = glm.vec2(self.zoom_pos)
        inverted_zoom_pos = inverted_zoom_pos * glm.vec2(selected_texture.size) / glm.vec2(image_size)
        inverted_zoom_pos -= self.zoom_pixel * 0.5
        inverted_zoom_pos.y = selected_texture.size[1] - inverted_zoom_pos.y - self.zoom_pixel
        uv0 = glm.vec2(glm.ivec2(inverted_zoom_pos)) / glm.vec2(selected_texture.size)
        uv1 = glm.vec2(glm.ivec2(inverted_zoom_pos + glm.vec2(self.zoom_pixel, self.zoom_pixel))) / glm.vec2(
            selected_texture.size
        )
        origin = imgui.get_cursor_screen_pos()

        imgui.begin_group()
        imgui.image(
            selected_texture.glo,
            (self.PREVIEW_SIZE, self.PREVIEW_SIZE),
            uv0=ImVec2(uv0.x, uv1.y),
            uv1=ImVec2(uv1.x, uv0.y),
        )
        offset = self.PREVIEW_SIZE / self.zoom_pixel * (self.zoom_pixel // 2)
        draw_list.add_rect(
            ImVec2(origin.x + offset, origin.y + offset),
            ImVec2(
                origin.x + offset + self.PREVIEW_SIZE / self.zoom_pixel,
                origin.y + offset + self.PREVIEW_SIZE / self.zoom_pixel,
            ),
            col_red,
            thickness=0.1,
        )
        imgui.same_line()
        _, self.zoom_pixel = imgui.v_slider_int(
            "##",
            ImVec2(20, self.PREVIEW_SIZE),
            self.zoom_pixel,
            1,
            self.PREVIEW_SIZE,
        )
        imgui.end_group()

        _, self.filter_min, self.filter_max = imgui.drag_float_range2(
            "##", self.filter_min, self.filter_max, 0.01, 0, 1000
        )
        imgui.same_line()
        if imgui.button("Reset"):
            self.filter_min = 0.0
            self.filter_max = 1.0
        imgui.end_group()
        imgui.end()


class SettingsViewer:
    def __init__(self) -> None:
        self.stop_frame_counter = False
        self.buffers: dict[str, ScrollingBuffer] = defaultdict(ScrollingBuffer)
        self.current_frame = 0
        self.total_buffer = ScrollingBuffer()

    def render(self, timings: dict[str, int]) -> None:
        total_time = 0
        for name, time in timings.items():
            self.buffers[name].add_point(self.current_frame, time * 0.000001)
            if name != "frame time":
                total_time += time
        self.total_buffer.add_point(self.current_frame, total_time * 0.000001)

        imgui.begin("Profiler", p_open=True)
        _, self.stop_frame_counter = imgui.checkbox("Stop Framecounter", self.stop_frame_counter)

        flags = implot.AxisFlags_.no_tick_labels
        if implot.begin_plot("##Scrolling", size=(-1, imgui.get_text_line_height() * 20)):
            implot.setup_axes("", "ms", flags, implot.AxisFlags_.range_fit)  # type:ignore[arg-type]
            implot.setup_axis_limits(
                implot.ImAxis_.x1,  # type:ignore[arg-type]
                self.current_frame - 100,
                self.current_frame,
                implot.Cond_.always,  # type:ignore[arg-type, attr-defined]
            )
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, 30)  # type:ignore[arg-type]
            xs1, ys1 = self.total_buffer.get_data()
            implot.plot_shaded("total time", xs1, ys1)
            for name, buffer in self.buffers.items():
                xs1, ys1 = buffer.get_data()
                implot.plot_line(name, xs1, ys1)
            implot.end_plot()
        imgui.end()
        self.current_frame += 1


@lru_cache(maxsize=1024)
def folder_structure(path: Path) -> list[Path]:
    folder_files = list(path.iterdir())
    folder_files.sort(key=lambda x: not x.is_dir())
    return folder_files


class ObjectsViewer:
    MODEL_DIR = Path("./resources/models/")

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.selected_object_state: tuple[str, int] | None = None

    def draw_model_file_tree(self, path: Path) -> None:
        for item in folder_structure(path):
            if item.is_dir():
                if imgui.tree_node(str(item.name)):
                    self.draw_model_file_tree(item)
                    imgui.tree_pop()
            elif item.suffix.lower() == ".txt":
                clicked, _ = imgui.selectable(str(item.name), p_selected=False)
                if clicked:
                    self.scene.add_voxel_object(VoxelObject(model=parse_model(item)))

    def render(self) -> None:  # noqa: C901, PLR0912
        if imgui.begin("Objects", p_open=True):
            if imgui.begin_child("object_list", size=(200, 0)):
                if imgui.collapsing_header(f"Voxos ({len(self.scene.voxel_objects)})"):
                    for i, obj in enumerate(self.scene.voxel_objects):
                        clicked, _ = imgui.selectable(obj.name, self.selected_object_state == ("Voxos", i))
                        if clicked:
                            self.selected_object_state = ("Voxos", i)
                if imgui.collapsing_header(f"Lights ({len(self.scene.lights)})"):
                    for i, light in enumerate(self.scene.lights):
                        clicked, _ = imgui.selectable(light.name, self.selected_object_state == ("Lights", i))
                        if clicked:
                            self.selected_object_state = ("Lights", i)
                    if imgui.button("Add Light", size=(0, 0)):
                        self.scene.add_light(Light())
                if imgui.collapsing_header(f"Suns ({len(self.scene.suns)})"):
                    for i, sun in enumerate(self.scene.suns):
                        clicked, _ = imgui.selectable(sun.name, self.selected_object_state == ("Suns", i))
                        if clicked:
                            self.selected_object_state = ("Suns", i)
                if imgui.collapsing_header("Models"):
                    self.draw_model_file_tree(self.MODEL_DIR)
            imgui.end_child()

            imgui.same_line()

            if imgui.begin_child("properties"):
                if self.selected_object:
                    _, self.selected_object.visible = imgui.checkbox("Visible", self.selected_object.visible)
                    imgui.separator_text("Transform")
                    t = self.selected_object.translation
                    _, new_trans = imgui.drag_float3(
                        "translation", t.to_list(), v_speed=1, v_min=-1000, v_max=1000, format="%.1f"
                    )
                    self.selected_object.translation = glm.vec3(new_trans)

                    r = self.selected_object.rotation
                    euler = cast("glm.vec3", glm.degrees(glm.eulerAngles(r)))
                    _, new_rot = imgui.drag_float3("rotation", euler.to_list(), v_speed=45, v_min=-360, v_max=360)
                    self.selected_object.rotation = glm.quat(glm.radians(glm.vec3(new_rot)))

                    s = self.selected_object.scale
                    _, new_scale = imgui.drag_float3("scale", s.to_list(), v_speed=0.1, v_min=-10, v_max=10)
                    self.selected_object.scale = glm.vec3(new_scale)

                    if isinstance(self.selected_object, VoxelObject):
                        imgui.separator_text("Dimensions")
                        dim = self.selected_object.model.opengl_dimensions
                        _, _ = imgui.drag_float3("##", list(dim), v_speed=0.0, v_min=-10, v_max=10, format="%.0f")

                    if isinstance(self.selected_object, Light):
                        imgui.separator_text("Light")
                        _, new_col = imgui.color_edit3("color", self.selected_object.color.to_list())
                        self.selected_object.color = glm.vec3(new_col)

                        _, self.selected_object.intensity = imgui.slider_float(
                            "intensity",
                            self.selected_object.intensity,
                            v_min=1.0,
                            v_max=20_000,
                            format="%.0f",
                        )
                        _, self.selected_object.radius = imgui.slider_float(
                            "radius",
                            self.selected_object.radius,
                            v_min=0.1,
                            v_max=20,
                            format="%.1f",
                        )

                else:
                    imgui.text("No Object selected.")
            imgui.end_child()

        imgui.end()

    @property
    def selected_object(self) -> Object | None:
        if self.selected_object_state is None:
            return None
        obj_type, index = self.selected_object_state
        if obj_type == "Suns":
            return self.scene.suns[index]
        if obj_type == "Lights":
            return self.scene.lights[index]
        if obj_type == "Voxos":
            return self.scene.voxel_objects[index]
        return None


def imgui_matrix(name: str, values: tuple[float], v_speed: float) -> None:
    if len(values) % 4 == 0:
        for row in range(len(values) // 4):
            imgui.drag_float4(f"{name}_{row}", list(values[row * 4 : row * 4 + 4]), v_speed=v_speed)
    else:
        assert len(values) % 3 == 0, len(values)
        for row in range(len(values) // 3):
            imgui.drag_float4(f"{name}_{row}", list(values[row * 3 : row * 3 + 3]), v_speed=v_speed)


class ShaderViewer:
    def __init__(self, shaders: list[Program]) -> None:
        self.shaders = shaders
        self.selected_shader_index: int = 0

    def render(self) -> None:  # noqa: C901, PLR0912
        if imgui.begin("Shaders", p_open=True):
            if imgui.begin_child("shader_list", size=(200, 0)):
                seen = set()
                for i, shader in enumerate(self.shaders):
                    assert shader.label
                    name = shader.label
                    while name in seen:
                        name += "#"
                    clicked, _ = imgui.selectable(name, self.selected_shader_index == i)
                    if clicked:
                        self.selected_shader_index = i
                    seen.add(name)
            imgui.end_child()

            imgui.same_line()

            if imgui.begin_child("properties"):
                for uniform_name in self.selected_shader:
                    uniform = self.selected_shader[uniform_name]
                    if not isinstance(uniform, Uniform):
                        continue
                    imgui.separator_text(uniform_name)
                    if uniform.dimension == 1:
                        if type(uniform.value) is int:
                            _, uniform.value = imgui.drag_int(f"##{uniform_name}", uniform.value, v_speed=1)
                        elif type(uniform.value) is float:
                            _, uniform.value = imgui.drag_float(f"##{uniform_name}", uniform.value, v_speed=0.1)
                        else:
                            raise NotImplementedError
                    elif uniform.dimension == 3:
                        imgui.drag_float3(f"##{uniform_name}", uniform.value, v_speed=0.1)
                    elif uniform.dimension == 4:
                        imgui.drag_float4(f"##{uniform_name}", uniform.value, v_speed=0.1)
                    elif uniform.dimension == 16:
                        imgui_matrix(f"##{uniform_name}", uniform.value, v_speed=0.1)
                    else:
                        raise NotImplementedError
            imgui.end_child()

        imgui.end()

    @property
    def selected_shader(self) -> Program:
        return self.shaders[self.selected_shader_index]


class DebugView(ModernglWindowRenderer):
    def __init__(self, window: WindowConfig, scene: Scene, textures: list[Texture], shaders: list[Program]) -> None:
        self.profiler = Profiler(window.ctx)
        imgui.create_context()
        implot.create_context()
        super().__init__(window.wnd)  # type:ignore[no-untyped-call]
        for texture in textures:
            self.register_texture(texture)
        self.texture_viewer = TextureViewer(textures, window)
        self.register_texture(self.texture_viewer.preview_texture)
        self.objects_viewer = ObjectsViewer(scene)
        self.shader_viewer = ShaderViewer(shaders)
        self.settings = SettingsViewer()

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard

    @property
    def is_frame_counter_stopped(self) -> bool:
        return self.settings.stop_frame_counter

    def render_debug(self, global_frame_counter: int, frametime: float) -> None:
        imgui.new_frame()
        self.texture_viewer.render()
        self.objects_viewer.render()
        self.shader_viewer.render()
        self.settings.render(self.profiler.all_timings(global_frame_counter, frametime))
        imgui.render()

        selected_texture = self.texture_viewer.textures[self.texture_viewer.selected_texture]
        prev_filter = selected_texture.filter
        selected_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        super().render(imgui.get_draw_data())
        selected_texture.filter = prev_filter
