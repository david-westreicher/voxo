import argparse
from pathlib import Path

import moderngl_window

from .main import VoxoWindow
from .model import parse_xml_level
from .objects import World


def convert_td_to_level(input_level: Path, output_level: Path | None = None) -> None:
    if output_level is None:
        output_level = input_level.parent.parent / input_level.parent.with_suffix(".lvl").name
    vox_models = list(parse_xml_level(input_level))
    World.from_vox_models(vox_models).write(output_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="voxo")
    parser.add_argument("--convert_td_to_level", type=Path, help="Convert xml-td to level format", metavar="FILE")
    parser.add_argument("--output_level", type=Path, help="Output level file", metavar="FILE", default=None)
    args = parser.parse_args()
    if args.convert_td_to_level:
        convert_td_to_level(args.convert_td_to_level, args.output_level)
    else:
        moderngl_window.run_window_config(VoxoWindow)
