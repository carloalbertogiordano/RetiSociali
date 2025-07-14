from pathlib import Path
import json


def analyze_and_average(collected_results_dir, exp_id, num_exec, suffix):
    """
    Analizza i risultati su più esecuzioni e plottane la media del numero di nodi influenzati per step.
    """
    all_runs_data = []

    for i in range(num_exec):
        file_path = Path(f"{collected_results_dir}/{exp_id}{i}/{exp_id}{i}_{suffix}")
        if file_path.exists():
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if data and isinstance(data[0], list):
                        all_runs_data.append(data[0])  # estrai il primo livello (lista di step)
                    else:
                        print(f"⚠️ Dati non nel formato atteso in: {file_path}")
                except Exception as e:
                    print(f"⚠️ Errore nel leggere {file_path}: {e}")
        else:
            print(f"⚠️ File non trovato: {file_path}")

    # Se non ci sono dati validi, termina
    if not all_runs_data:
        print("❌ Nessun dato valido trovato. Interrotto.")
        return

    print(f"Run data: {all_runs_data}")

    sum = 0.0
    i = 0
    for execution in all_runs_data:
        cascade = len(execution[-1])
        ss = len(execution[0])

        sum += cascade - ss
        i += 1

    return sum/i
