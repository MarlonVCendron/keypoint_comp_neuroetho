import h5py
import os

def prevent_overwrite_error(project_dir, model_name, recording_names):
    overwrite = False
    results_path = os.path.join(project_dir, model_name, "results.h5")
    if os.path.exists(results_path):
        with h5py.File(results_path, "r+") as f:
            for recording_name in recording_names:
                if recording_name in f:
                    if not overwrite:
                        overwrite = ask_to_overwrite(recording_name, results_path)
                        if overwrite:
                            create_backup_results(results_path)
                    if overwrite:
                        del f[recording_name]

def create_backup_results(results_path):
    if os.path.exists(results_path):
        os.rename(results_path, results_path + ".old")

def ask_to_overwrite(recording_name, results_path):
    answer = input(f"{recording_name} já existe em {results_path}. Deseja sobrescrever? (y/n)")
    if answer == "y":
        return True
    else:
        print("Se não sobreescrever vai dar erro!")
        exit()
