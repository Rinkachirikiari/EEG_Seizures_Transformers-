import mne
import numpy as np
import matplotlib.pyplot as plt


# Charger le fichier EDF
raw = mne.io.read_raw_edf('chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0/chb07/chb07_01.edf')

# Afficher les informations du fichier
print(raw.info)

# Afficher les premières secondes des données
print(raw.get_data(start=0, stop=10))  # Afficher les données des 10 premières secondes

# Extraire les données et les temps
data, times = raw[:, :]

raw.plot(n_channels=len(raw.ch_names), title='Données de chaque canal', scalings='auto')


# Initialisation des listes pour stocker les informations
seizure_presence = {}

# Chemin d'accès au fichier
file_path = "chb04-summary.txt"

# Ouvrir et lire le fichier
is_seizure = False
all_files_str = []
seizure_start = 0
with open(file_path, "r") as file:
    # Lire chaque ligne du fichier
    for line in file:
        # Traiter la ligne actuelle
        if line.strip():
            if "File Name:" in line:
              start = len("File Name: ")
              all_files_str.append(str(line[start:len(line)-1]))
            # elif "Number of Seizures in File:" in line:
            #   start = len("Number of Seizures in File: ")
            #   number_of_seizure = int(line[start:len(line)-2])
            #   if number_of_seizure == 0:
            #     is_seizure = False
            #   else:
            #     is_seizure = True
            elif ("Seizure" in line) and ("Start Time: " in line):
              if line[len("Seizure S")-1] == "S":
                start = len("Seizure Start Time: ")
              else:
                start = len("Seizure 1 Start Time: ")
              end = len(" seconds")+1
              seizure_start = int(line[start:len(line)-end])
            elif ("Seizure"in line) and ("End Time: " in line):
              if line[len("Seizure E")-1] == "E":
                start = len("Seizure End Time: ")
              else:
                start = len("Seizure 1 End Time: ")
              end = len(" seconds")+1
              seizure_end = int(line[start:len(line)-end])
              if not all_files_str[len(all_files_str)-1] in seizure_presence.keys():
                seizure_presence[all_files_str[len(all_files_str)-1]] = []
              seizure_presence[all_files_str[len(all_files_str)-1]].append((seizure_start, seizure_end))


print(all_files_str)
print(seizure_presence)


def seperate_data_intervals(file_str, seizure_presence):
  # Durée de chaque intervalle en secondes (5 minutes)
  interval_duration = 5 * 60

  raw = mne.io.read_raw_edf(file_str)
  data, times = raw[:, :]
  total_duration = raw.times[-1] # en secondes

  # Nombre total d'intervalle de 10 minutes
  num_intervals = int(total_duration / interval_duration)
  labels = []
  data = []

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
      data.append(interval_data)
      if file_str in seizure_presence.keys():
        is_seizure = False
        for start_seizure, end_seizure in seizure_presence[file_str]:
          if (start_seizure >= start_time and start_seizure <= end_time) or (end_seizure >= start_time and end_seizure <= end_time) or (start_seizure <= start_time and end_seizure >= end_time):
            is_seizure = True
            break
        if is_seizure:
          labels.append(1)
        else:
          labels.append(0)
      else:
        labels.append(0)
  return data, labels


data = []
labels = []
taille = 0
for file in all_files_str:
  interval_data, label = seperate_data_intervals(file, seizure_presence)
  for i in range(len(interval_data)):
    data.append(interval_data[i])
    labels.append(label[i])


print(len(data))
print(len(data[0]))
print(len(data[0][0]))

