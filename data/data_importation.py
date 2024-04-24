import os
import mne


def import_data():
    file_path = "../chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0"
    save_path = "saved_fif_files"  # Define where to save the .fif files

    # Create the directory for saving .fif files if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Dictionary to store MNE data by subfolder
    donnees_par_sous_dossier = {}

    # Traverse subfolders
    for nom_sous_dossier in os.listdir(file_path):
        chemin_sous_dossier = os.path.join(file_path, nom_sous_dossier)

        if os.path.isdir(chemin_sous_dossier):
            donnees_par_sous_dossier[nom_sous_dossier] = []
            patient_folder_path = os.path.join(save_path, nom_sous_dossier)
            os.makedirs(patient_folder_path, exist_ok=True)

            for nom_fichier in os.listdir(chemin_sous_dossier):
                if nom_fichier.endswith('.edf'):
                    chemin_fichier = os.path.join(chemin_sous_dossier, nom_fichier)
                    donnees_mne = mne.io.read_raw_edf(chemin_fichier, preload=True, verbose=False)
                    donnees_mne.set_meas_date(None)  # Reset the measurement date to avoid issues
                    donnees_par_sous_dossier[nom_sous_dossier].append(donnees_mne)

                    # Save the read data into .fif format
                    save_file_path = os.path.join(patient_folder_path, f"{nom_fichier[:-4]}.fif")
                    donnees_mne.save(save_file_path, overwrite=True)

    # Display information about the MNE data
    for nom_sous_dossier, donnees_mne in donnees_par_sous_dossier.items():
        print(f"Sujet {nom_sous_dossier} : {len(donnees_mne)} fichiers .edf")


if __name__ == "__main__":
    import_data()
