from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from pyglm import glm

from .vox_parser import SimplifiedVoxFile, VoxModel


def parse_vec2(pos_attribute: dict[str, str], field: str) -> glm.vec2:
    pos = [float(e) for e in pos_attribute[field].split()] if field in pos_attribute else [0.0, 0.0]
    return glm.vec2(pos)


def parse_vec3(pos_attribute: dict[str, str], field: str) -> glm.vec3:
    pos = [float(e) for e in pos_attribute[field].split()] if field in pos_attribute else [0.0, 0.0, 0.0]
    return glm.vec3(pos)


@dataclass
class VoxLight:
    name: str
    light_type: str
    reach: float = 0.0
    unshadowed: float = 0.0
    glare: float = 0.0
    scale: float = 0.0
    angle: float = 0.0
    penumbra: float = 0.0
    light_size: float = 0.1
    translation: glm.vec3 = field(default_factory=lambda: glm.vec3(0))
    rotation: glm.quat = field(default_factory=glm.quat)
    color: glm.vec3 = field(default_factory=lambda: glm.vec3(0))
    size: glm.vec2 = field(default_factory=lambda: glm.vec2(0))


def parse_light(xml_light: ET.Element, parent_shape_name: str) -> Iterator[VoxLight]:
    light = VoxLight(
        name=parent_shape_name,
        light_type=xml_light.attrib.get("type", "sphere"),
        translation=parse_vec3(xml_light.attrib, field="pos") * 10.0 - glm.vec3(1.0, 0.0, 0.0),
        rotation=glm.quat(glm.radians(parse_vec3(xml_light.attrib, field="rot"))),
        color=parse_vec3(xml_light.attrib, field="color"),
        reach=float(xml_light.attrib.get("reach", 0.0)),
        unshadowed=float(xml_light.attrib.get("unshadowed", 0.0)),
        glare=float(xml_light.attrib.get("glare", 0.0)),
        scale=float(xml_light.attrib.get("scale", 1.0)),
        angle=float(xml_light.attrib.get("angle", 90.0)),
        penumbra=float(xml_light.attrib.get("penumbra", 10.0)),
    )
    if light.light_type in ["area", "capsule"]:
        light.size = parse_vec2(xml_light.attrib, field="size")
    else:
        light.light_size = float(xml_light.attrib.get("size", 0.1))
    yield light


def parse_vox_object(
    xml_vox: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    pos = parse_vec3(xml_vox.attrib, field="pos") * 10.0
    rot = glm.quat(glm.radians(parse_vec3(xml_vox.attrib, field="rot")))
    file = xml_vox.attrib["file"]
    shape = xml_vox.attrib["object"]
    if file not in vox_file_map:
        full_vox_file_path = level_xml.parent / Path(file).relative_to("MOD")
        vox_file_obj = SimplifiedVoxFile.from_file(full_vox_file_path)
        vox_file_map[file] = vox_file_obj
    vox_model = vox_file_map[file].get_model(shape).copy()
    vox_model.translation = pos
    vox_model.rotation = rot
    vox_model.shape_name = shape
    for xml_wheel in xml_vox.findall("wheel"):
        for vox_obj in parse_wheel(xml_wheel, vox_file_map, level_xml):
            vox_obj.rotation = rot * vox_obj.rotation
            vox_obj.translation = pos + rot * vox_obj.translation
            yield vox_obj
    for xml_light in xml_vox.findall("light"):
        for light in parse_light(xml_light, parent_shape_name=shape):
            light.rotation = rot * light.rotation
            light.translation = pos + rot * light.translation
            yield light
    yield vox_model


def parse_compound(
    xml_compound: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    compound_pos = parse_vec3(xml_compound.attrib, field="pos") * 10.0
    compound_rot = glm.quat(glm.radians(parse_vec3(xml_compound.attrib, field="rot")))
    for xml_vox in xml_compound.findall("vox"):
        for vox_obj in parse_vox_object(xml_vox, vox_file_map, level_xml):
            vox_obj.rotation *= compound_rot
            vox_obj.translation = compound_pos + compound_rot * vox_obj.translation
            yield vox_obj


def parse_world_body(
    xml_world_body: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    # TODO(david): parse voxbox

    for xml_vox in xml_world_body.findall("vox"):
        yield from parse_vox_object(xml_vox, vox_file_map, level_xml)

    for xml_compound in xml_world_body.findall("compound"):
        yield from parse_compound(xml_compound, vox_file_map, level_xml)


def parse_props(
    xml_props: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    for xml_body in xml_props.findall("body"):
        body_pos = parse_vec3(xml_body.attrib, field="pos") * 10.0
        body_rot = glm.quat(glm.radians(parse_vec3(xml_body.attrib, field="rot")))
        for xml_vox in xml_body.findall(".//vox"):
            for vox_obj in parse_vox_object(xml_vox, vox_file_map, level_xml):
                vox_obj.rotation = body_rot * vox_obj.rotation
                vox_obj.translation = body_pos + body_rot * vox_obj.translation
                yield vox_obj


def parse_wheel(
    xml_wheel: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    wheel_pos = parse_vec3(xml_wheel.attrib, field="pos") * 10.0
    wheel_rot = glm.quat(glm.radians(parse_vec3(xml_wheel.attrib, field="rot")))
    for xml_vox in xml_wheel.findall("vox"):
        for vox_obj in parse_vox_object(xml_vox, vox_file_map, level_xml):
            vox_obj.rotation = wheel_rot * vox_obj.rotation
            vox_obj.translation = wheel_pos + wheel_rot * vox_obj.translation
            yield vox_obj


def parse_body(
    xml_body: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    body_pos = parse_vec3(xml_body.attrib, field="pos") * 10.0
    body_rot = glm.quat(glm.radians(parse_vec3(xml_body.attrib, field="rot")))
    for xml_vox in xml_body.findall("vox"):
        for vox_obj in parse_vox_object(xml_vox, vox_file_map, level_xml):
            vox_obj.rotation = body_rot * vox_obj.rotation
            vox_obj.translation = body_pos + body_rot * vox_obj.translation
            yield vox_obj
    for xml_wheel in xml_body.findall("wheel"):
        for vox_obj in parse_wheel(xml_wheel, vox_file_map, level_xml):
            vox_obj.rotation = body_rot * vox_obj.rotation
            vox_obj.translation = body_pos + body_rot * vox_obj.translation
            yield vox_obj


def parse_vehicles(
    xml_props: ET.Element,
    vox_file_map: dict[str, SimplifiedVoxFile],
    level_xml: Path,
) -> Iterator[VoxModel | VoxLight]:
    for xml_vehicle in xml_props.findall("vehicle"):
        vehicle_pos = parse_vec3(xml_vehicle.attrib, field="pos") * 10.0
        vehicle_rot = glm.quat(glm.radians(parse_vec3(xml_vehicle.attrib, field="rot")))
        for xml_body in xml_vehicle.findall("body"):
            for vox_obj in parse_body(xml_body, vox_file_map, level_xml):
                vox_obj.rotation = vehicle_rot * vox_obj.rotation
                vox_obj.translation = vehicle_pos + vehicle_rot * vox_obj.translation
                yield vox_obj


def parse_xml_level(level_xml: Path) -> Iterator[VoxModel | VoxLight]:
    tree = ET.parse(level_xml)  # noqa: S314
    root = tree.getroot()
    vox_file_map: dict[str, SimplifiedVoxFile] = {}

    world_body = root.find(".//group[@name='World Body']")
    if world_body:
        yield from parse_world_body(world_body, vox_file_map, level_xml)

    props = root.find(".//group[@name='Props']")
    if props:
        yield from parse_props(props, vox_file_map, level_xml)

    vehicles = root.find(".//group[@name='Vehicles']")
    if vehicles:
        yield from parse_vehicles(vehicles, vox_file_map, level_xml)
