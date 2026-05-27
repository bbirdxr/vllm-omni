"""一键顺序跑完所有 lecture.

用法:
    python tools/lectures/run_all.py
    python tools/lectures/run_all.py --skip 04         # 跳过某节
    python tools/lectures/run_all.py --only 02 03      # 只跑这几节
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

LECTURES = [
    ("01", "01_config_registration"),
    ("02", "02_pipeline_resolution"),
    ("03", "03_model_registry"),
    ("04", "04_stage_configs"),
    ("05", "05_stage_spawn_and_api_routing"),
    ("06", "06_chat_to_engine_prompt"),
    ("07", "07_request_lifecycle"),
    ("08", "08_llm2tts_bridge"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="lecture ids to skip")
    parser.add_argument("--only", nargs="*", default=[], help="run ONLY these ids")
    args = parser.parse_args()

    selected = [(i, m) for i, m in LECTURES if i not in args.skip]
    if args.only:
        selected = [(i, m) for i, m in selected if i in args.only]

    if not selected:
        print("no lectures selected")
        return

    print(f"将依次运行: {[i for i, _ in selected]}")
    for lecture_id, module_name in selected:
        print()
        print("#" * 70)
        print(f"# Lecture {lecture_id}  ({module_name}.py)")
        print("#" * 70)
        mod = importlib.import_module(module_name)
        mod.main()


if __name__ == "__main__":
    main()
