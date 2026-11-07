#!/usr/bin/env python3
"""Build the family reward chart data from the recorded prime episodes."""
import csv
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "alias_acquisition": "Alias acquisition",
    "output_code_remapping": "Output-code remapping",
    "lexical_inhibition": "Lexical inhibition",
    "terminology_preference": "Terminology preference",
    "default_response_language": "Default response language",
    "structured_response_default": "Structured response default",
    "answer_length_control": "Answer-length control",
    "factual_update": "Local factual update",
    "selective_factual_withholding": "Selective factual withholding",
    "resistance_to_false_suggestions": "Resistance to false suggestions",
}
INPUTS = {
    "6.1-terra": ("gpt-6.1-terra", "results/ten-families-v022/gemma-3-4b-it-prime.csv"),
    "6.1-sol (low)": ("gpt-6.1-sol", "results/ten-families-v022-sol/gemma-3-4b-it-prime.csv"),
    "Astra": ("gpt-6-astra", "results/ten-families-v022-astra/gemma-3-4b-it-prime.csv"),
    "GLM 5.3": ("glm-5.3", "results/ten-families-v022-glm/gemma-3-4b-it-prime.csv"),
    "DeepSeek V4.1 Flash": ("deepseek-v4.1-flash-1017", "results/ten-families-v022-deepseek/gemma-3-4b-it-prime.csv"),
    "Nemotron 3 Ultra": ("nemotron-3-ultra-550b-a55b", "results/ten-families-v022-nemotron/gemma-3-4b-it-prime.csv"),
    "6.1-sol (medium)": ("gpt-6.1-sol", "results/ten-families-v022-sol-medium/gemma-3-4b-it-prime.csv"),
    "6.1-sol (high)": ("gpt-6.1-sol", "results/ten-families-v022-sol-high/gemma-3-4b-it-prime.csv"),
    "5.6-sol (low)": ("gpt-5.6-sol", "results/ten-families-v022-56-sol/gemma-3-4b-it-prime.csv"),
    "5.6-terra (low)": ("gpt-5.6-terra", "results/ten-families-v022-56-terra/gemma-3-4b-it-prime.csv"),
    "Inkling 2 Small (low)": ("thinkingmachines/Inkling-2-Small:peft:262144", "results/ten-families-v022-inkling-2-small-low/gemma-3-4b-it-prime.csv"),
}
MODULATION = {
    "alias_acquisition": "stimulatory", "structured_response_default": "stimulatory",
    "lexical_inhibition": "inhibitory", "selective_factual_withholding": "inhibitory",
    "resistance_to_false_suggestions": "inhibitory", "output_code_remapping": "modulation",
    "terminology_preference": "modulation", "default_response_language": "modulation",
    "answer_length_control": "modulation", "factual_update": "modulation",
}


def means(path, editor):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            expected = {"editor_model": editor,
                        "model_id": "google/gemma-3-4b-it", "family_version": "0.2.2",
                        "revision": "093f9f388b31de276ce2de164bdc2081324b9767"}
            for key, value in expected.items():
                if row[key] != value:
                    raise ValueError(f"{path}: expected {key}={value}, got {row[key]}")
            if "run_type" in row:
                if row["run_type"] != "prime":
                    raise ValueError(f"{path}: expected prime run_type")
            elif not (editor in {"gpt-5.6-sol", "gpt-5.6-terra"} and row.get("reasoning_effort") == "low" and row.get("native_episode_id")):
                raise ValueError(f"{path}: unrecognised episode export schema")
            reward = float(row["reward"])
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError(f"{path}: invalid reward {reward}")
            groups[row["family"]].append(reward)
    if set(groups) != set(FAMILIES) or any(len(v) != 3 for v in groups.values()):
        raise ValueError(f"{path}: expected three episodes in each of ten families")
    return {key: sum(groups[key]) / 3 for key in FAMILIES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", action="append", nargs=3, metavar=("LABEL", "MODEL_ID", "CSV"),
                        help="Editor label, model ID and CSV. Repeat for each editor.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/data/family-rewards.json")
    args = parser.parse_args()
    editors = args.editor or [(name, editor, path) for name, (editor, path) in INPUTS.items()]
    if len({name for name, _, _ in editors}) != len(editors):
        raise ValueError("Editor labels must be unique")
    scores = {name: means(ROOT / path, editor) for name, editor, path in editors}
    output = {"target": "Gemma 3 4B IT", "family_version": "0.2.2", "episodes_per_family": 3,
              "families": [{"id": key, "label": label, "modulation": MODULATION[key],
                            "means": {name: scores[name][key] for name in scores}}
                           for key, label in FAMILIES.items()]}
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()


