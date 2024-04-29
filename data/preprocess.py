import gc
import json
import multiprocessing
import os
import pickle

import h5py
import mne
import numpy as np


def separate_data_intervals(file_str, seizure_presence, path):
    # Durée de chaque intervalle en secondes (2 minutes)
    interval_duration = 10

    raw = mne.io.read_raw_edf(path + file_str, preload=False)
    # data, times = raw[:, :]
    total_duration = raw.times[-1]  # en secondes

    # Nombre total d'intervalle de 10 minutes
    num_intervals = int(total_duration / interval_duration)
    data = []
    labels = []

    count_0 = 0
    # Diviser les données en intervalles de 10 minutes
    for i in range(num_intervals):
        # Calculer le temps de début et de fin de chaque intervalle
        start_time = i * interval_duration
        end_time = (i + 1) * interval_duration

        # Convertir le temps en indice
        start_idx = raw.time_as_index(start_time)
        end_idx = raw.time_as_index(end_time)

        # Extraire les données de l'intervalle
        interval_data, interval_times = raw[:, start_idx:end_idx]

        seizure_times = seizure_presence.get(file_str, [])

        current_labels = any((start <= end_time and start >= start_time) or
                     (end >= start_time and end <= end_time) or
                     (start <= start_time and end >= end_time)
                     for start, end in seizure_times)
        if (current_labels == 0 and count_0 % 10 == 0) or current_labels == 1:
            labels.append(current_labels)
            data.append(interval_data.astype(np.float32))
        count_0 += 1


    del raw
    gc.collect()
    return data, labels


def read_summary(dossier_parent):
    # Initialisation des listes pour stocker les informations
    seizure_presence = {}

    # Ouvrir et lire le fichier
    is_seizure = False
    all_files_str = []
    seizure_start = 0
    with open(dossier_parent, "r") as file:
        # Lire chaque ligne du fichier
        for line in file:
            # Traiter la ligne actuelle
            if line.strip():
                if "File Name:" in line:
                    start = len("File Name: ")
                    all_files_str.append(str(line[start:len(line) - 1]))
                elif ("Seizure" in line) and ("Start Time: " in line):
                    if line[len("Seizure S") - 1] == "S":
                        start = len("Seizure Start Time: ")
                    else:
                        start = len("Seizure 1 Start Time: ")
                    end = len(" seconds") + 1
                    seizure_start = int(line[start:len(line) - end])
                elif ("Seizure" in line) and ("End Time: " in line):
                    if line[len("Seizure E") - 1] == "E":
                        start = len("Seizure End Time: ")
                    else:
                        start = len("Seizure 1 End Time: ")
                    end = len(" seconds") + 1
                    seizure_end = int(line[start:len(line) - end])
                    if not all_files_str[len(all_files_str) - 1] in seizure_presence.keys():
                        seizure_presence[all_files_str[len(all_files_str) - 1]] = []
                    seizure_presence[all_files_str[len(all_files_str) - 1]].append((seizure_start, seizure_end))
    print("Seizure presence init", seizure_presence)
    return seizure_presence, all_files_str


def load_data_for_patient(dossier_parent, patient):
    path = dossier_parent + "/" + patient + "/"
    seizure_presence, all_files_str = read_summary(path + patient + "-summary.txt")
    patient_data = {'data': [], 'labels': []}

    for file in all_files_str:
        try:
            print("Reading file : ", file)
            interval_data, labels = separate_data_intervals(file, seizure_presence, path)
            print("End of file")

            patient_data['data'].extend(interval_data)
            patient_data['labels'].extend(labels)

        except Exception as e:
            print("Error while reading file : ", file)
            print(e)

    patient_data['data'] = np.array(patient_data['data'])
    patient_data['labels'] = np.array(patient_data['labels'], dtype=int)

    return patient_data


def save_patient_data_to_hdf5(file_path, patient_id, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open a new HDF5 file
    with h5py.File(file_path, 'w') as file:
        # Create a group for the patient
        patient_group = file.create_group(patient_id)

        # Iterate over data items (e.g., 'data', 'labels')
        for key in data:
            if isinstance(data[key], list):
                data[key] = np.array(data[key])

            patient_group.create_dataset(key, data=data[key])


def save_data(filename, patient_data):
    with open(filename, 'wb') as file:
        pickle.dump(patient_data, file)


def save_preprocess_data(dossier_parent, patients):
    for patient in patients:
        patient_data = load_data_for_patient(dossier_parent, patient)
        file_path = os.path.join('patients', f"under_sample_{patient}")
        save_data(file_path, patient_data)
        patient_data.clear()


if __name__ == "__main__":
    patients = ["chb01", "chb02", "chb03", "chb05", "chb06", "chb07", "chb08", "chb23", "chb24"]
    file_path = "../chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0"

    save_preprocess_data(file_path, patients)
