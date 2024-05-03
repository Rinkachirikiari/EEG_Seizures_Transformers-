import numpy as np


def segment_and_reconstruction(patient_eeg):
    nb_samples = patient_eeg.shape[0]
    segment_length = patient_eeg.shape[2]
    nb_segments = 20
    points_per_segments = segment_length // nb_segments
    augmented_eeg = np.empty_like(patient_eeg)
    for i in range(nb_samples):
        positive_segments = [patient_eeg[i, :, segment * points_per_segments:(segment + 1) * points_per_segments] for
                             segment in range(nb_segments)]
        np.random.shuffle(positive_segments)
        augmented_eeg[i] = np.concatenate(positive_segments, axis=1)
    return augmented_eeg


def augment_data(patient_datas, data_labels):
    label_ones = np.where(data_labels == 1)[0]
    patient_data_to_augment = patient_datas[label_ones]

    augmented_datas = []
    augmented_labels = []
    for i in range(10):
        augmented_eeg = segment_and_reconstruction(patient_data_to_augment)
        augmented_datas.append(augmented_eeg)
        augmented_labels.append(np.ones(augmented_eeg.shape[0], dtype=data_labels.dtype))

    augmented_datas_array = np.concatenate(augmented_datas, axis=0)
    augmented_labels_array = np.concatenate(augmented_labels, axis=0)

    patient_datas = np.concatenate([patient_datas, augmented_datas_array], axis=0)
    data_labels = np.concatenate([data_labels, augmented_labels_array], axis=0)

    indices = np.random.permutation(len(patient_datas))
    patient_datas = patient_datas[indices]
    data_labels = data_labels[indices]

    return patient_datas, data_labels
