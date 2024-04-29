import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE


def reshape_data(data):
    n_samples, dim1, dim2 = data.shape
    return data.reshape(n_samples, dim1 * dim2)


def down_sample(x_train_val, y_train_val):
    majority_indices = np.where(y_train_val == 0)[0]
    minority_indices = np.where(y_train_val == 1)[0]

    half_majority_size = len(majority_indices) // 10
    downsampled_majority_indices = np.random.choice(majority_indices, half_majority_size, replace=False)

    combined_indices = np.concatenate([downsampled_majority_indices, minority_indices])
    x_train_val_resampled = x_train_val[combined_indices]
    y_train_val_resampled = y_train_val[combined_indices]

    return x_train_val_resampled, y_train_val_resampled


def split_data(data, labels, test_size=0.2, val_size=0.25, random_state=42):
    x_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    #x_train_val_resampled, y_train_val_resampled = down_sample(x_train_val, y_train_val)

    #x_train_val_resampled_reshaped = reshape_data(x_train_val_resampled)
    #print(np.sum(y_train_val_resampled == 1))
    #smote = BorderlineSMOTE(kind='borderline-1', random_state=42)
    #x_train_val_res, y_train_val_res = smote.fit_resample(x_train_val_resampled_reshaped, y_train_val_resampled)
    #print(np.sum(y_train_val_res == 1))
    X_train, X_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
