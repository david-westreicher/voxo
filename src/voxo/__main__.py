import argparse
from pathlib import Path

import moderngl_window

from .main import VoxoWindow
from .model import parse_xml_level
from .objects import VoxelObject, World


def convert_td_to_level(input_world: Path) -> None:
    output_level = input_world.parent.parent / input_world.parent.with_suffix(".lvl").name
    vox_objs = [
        VoxelObject(
            name=obj.shape_name,
            model=obj.to_model(),
            rotation=obj.rotation,
            translation=obj.translation,
        )
        for obj in parse_xml_level(input_world)
    ]
    World().write(output_level, vox_objs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="voxo")
    parser.add_argument("--convert_td_to_level", type=Path, help="Convert xml-td to level format", metavar="FILE")
    args = parser.parse_args()
    if args.convert_td_to_level:
        convert_td_to_level(args.convert_td_to_level)
    else:
        moderngl_window.run_window_config(VoxoWindow)
