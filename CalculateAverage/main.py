import os
import shutil
import subprocess
import yaml
import json
from pathlib import Path
from calc_average import analyze_and_average

# --- Base experiment name ---
EXPERIMENT_ID = "experiment_"

# --- Paths ---
CONFIG_PATH = "base_conf.yaml"
UPDATED_CONFIG_PATH = "../config.yaml"
REMOTE_MAIN_PATH = "../main.py"
RESULTS_SRC_DIR = Path("../results/txt/")

# --- Number of executions ---
NUM_EXEC = 5

def update_yaml_config(original_path, updated_path, new_info_name):
    """Loads the YAML config, modifies the info_name, and writes it back."""
    with open(original_path, "r") as f:
        config = yaml.safe_load(f)

    config['graph']['info_name'] = new_info_name

    with open(updated_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"✅ Updated YAML with info_name: '{new_info_name}'")


def run_external_main(yaml_path):
    """Runs the external script using the provided YAML config in its own directory."""
    script_dir = os.path.dirname(REMOTE_MAIN_PATH)
    script_name = os.path.basename(REMOTE_MAIN_PATH)

    try:
        subprocess.run(
            ["python", script_name, os.path.basename(yaml_path)],
            check=True,
            cwd=script_dir
        )
    except subprocess.CalledProcessError as e:
        print("❌ Error while running external main:", e)
        exit(1)


def collect_and_move_results(experiment_name, dest_dir):
    """Moves result JSON files to a local experiment folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved_files = []

    for file in RESULTS_SRC_DIR.glob(f"{experiment_name}_*.json"):
        destination = dest_dir / file.name
        shutil.move(str(file), str(destination))
        moved_files.append(destination)

    return moved_files


def print_results(json_files):
    """Prints summary or content of each result JSON file."""
    for file in sorted(json_files):
        print(f"\n=== {file.name} ===")
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"⚠️  Failed to load {file}: {e}")

def delete_json_results(directory):
    """Deletes all JSON files in the given directory."""
    json_files = Path(directory).glob("*.json")
    for file in json_files:
        try:
            file.unlink()
            print(f"🗑️ Deleted: {file}")
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")



def main():

    suffixes = (
        "CSG_custom_f1.json", "CSG_degree_f1.json", "CSG_random_f1.json",
        "GENETIC_custom_none.json", "WTSS_custom_none.json",
        "CSG_custom_f2.json", "CSG_degree_f2.json", "CSG_random_f2.json",
        "GENETIC_degree_none.json", "WTSS_degree_none.json",
        "CSG_custom_f3.json", "CSG_degree_f3.json", "CSG_random_f3.json",
        "GENETIC_random_none.json", "WTSS_random_none.json"
    )

    collected_results_dir = Path("./collected_results")
    # Delete collected_results folder only before the first experiment
    if collected_results_dir.exists() and collected_results_dir.is_dir():
        print("🗑️ Clearing previous collected_results directory...")
        shutil.rmtree(collected_results_dir)

    for i in range(NUM_EXEC):
        current_experiment_id = f"{EXPERIMENT_ID}{i}"
        current_results_dir = Path(f"./collected_results/{current_experiment_id}/")

        delete_json_results(RESULTS_SRC_DIR)

        print(f"\n🔁 Running experiment: {current_experiment_id}")

        print("🔧 Updating YAML configuration...")
        update_yaml_config(CONFIG_PATH, UPDATED_CONFIG_PATH, current_experiment_id)

        print("🚀 Running external script...")
        run_external_main(UPDATED_CONFIG_PATH)

        print("📦 Collecting results...")
        json_files = collect_and_move_results(current_experiment_id, current_results_dir)

        if not json_files:
            print("⚠️  No result files found.")
        else:
            print(f"✅ Moved {len(json_files)} result files to {current_results_dir}")

    results = {}

    for suffix in suffixes:
        print(f"\n🔍 Analizzo: {suffix}")
        avg = analyze_and_average("./collected_results", EXPERIMENT_ID, NUM_EXEC, suffix)
        results[suffix] = avg

    # Stampa ordinata dei risultati
    print("\n📊 RISULTATI FINALI\n" + "=" * 50)
    for suffix, avg in sorted(results.items()):
        if avg is not None:
            print(f"📁 {suffix} ➜  ΔMedia Cascade - SS: {avg:.2f}")
        else:
            print(f"📁 {suffix} ➜  ⚠️  Nessun dato valido")

if __name__ == "__main__":
    main()
