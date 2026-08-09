import argparse
from pathlib import Path

import moderngl_window

from .main import VoxoWindow
from .model import VoxLight, VoxModel, parse_xml_level
from .model.level_parser import VoxWater
from .objects import World


def convert_td_to_level(input_level: Path, output_level: Path | None = None) -> None:
    if output_level is None:
        output_level = input_level.parent.parent / input_level.parent.with_suffix(".lvl").name
    print("Converting level", input_level, "to", output_level)  # noqa: T201
    vox_objects = list(parse_xml_level(input_level))
    vox_models = [obj for obj in vox_objects if type(obj) is VoxModel]
    vox_lights = [obj for obj in vox_objects if type(obj) is VoxLight]
    vox_waters = [obj for obj in vox_objects if type(obj) is VoxWater]
    World.from_vox_objects(vox_models, vox_lights, vox_waters).write(output_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="voxo")
    parser.add_argument("--convert_td_to_level", type=Path, help="Convert xml-td to level format", metavar="FILE")
    parser.add_argument("--output_level", type=Path, help="Output level file", metavar="FILE", default=None)
    args = parser.parse_args()
    if args.convert_td_to_level:
        convert_td_to_level(args.convert_td_to_level, args.output_level)
    else:
        moderngl_window.run_window_config(VoxoWindow)
