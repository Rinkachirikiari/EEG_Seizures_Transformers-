import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model.conformer import Conformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from data.load_data import load_data, check_memory_usage
from trainTest.split_data import split_data


def train_model(model, train_loader, validation_loader, device, criterion, num_epochs, patience, patient):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(optimizer, max_lr=0.0001, total_steps=num_epochs * len(train_loader))
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler()
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_correct = 0
        epoch_samples = 0

        all_targets = []
        all_predictions = []
        progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True)
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # with autocast():
            outputs = model(inputs)
            # y_hat = torch.argmax(outputs, dim=1).float()
            # y_hat = y_hat.requires_grad_()
            loss = criterion(outputs, targets)

            # scaler.step(optimizer)
            # scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            batch_loss = loss.item() * inputs.size(0)
            epoch_loss += batch_loss
            epoch_samples += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.set_postfix(loss=(epoch_loss / epoch_samples), accuracy=(total_correct / epoch_samples))
            progress_bar.update(1)

            loss.backward()
            optimizer.step()

        progress_bar.close()

        training_loss = epoch_loss / epoch_samples
        training_accuracy = total_correct / epoch_samples
        validation_loss, validation_accuracy, _ = validation(model, validation_loader, criterion, patient)

        history['train_loss'].append(training_loss)
        history['train_accuracy'].append(training_accuracy)
        history['val_loss'].append(validation_loss)
        history['val_accuracy'].append(validation_accuracy)

        conf_matrix = confusion_matrix(all_targets, all_predictions)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matrice confusion Train')
        plt.xlabel('Prédit')
        plt.ylabel('Target')
        plt.savefig(f'confusion_all_patients/matrice_confusion_train.png')
        plt.close()

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    return model, history


def validation(model, loader, criterion, patient):
    model.eval()
    validation_loss = 0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            # with autocast():
            #    outputs = model(inputs)
            # with autocast(enabled=False):
            #    loss = criterion(outputs.float(), targets.long())

            outputs = model(inputs)
            #y_hat = torch.argmax(outputs, dim=1).float()
            #y_hat = y_hat.requires_grad_()
            loss = criterion(outputs, targets)

            validation_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_validation_loss = validation_loss / total_samples
    validation_accuracy = correct_predictions / total_samples

    conf_matrix = confusion_matrix(all_targets, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Matrice confusion Validation')
    plt.xlabel('Prédit')
    plt.ylabel('Target')
    plt.savefig(f'confusion_all_patients/matrice_confusion_validation.png')
    plt.close()

    return avg_validation_loss, validation_accuracy, conf_matrix


def test_model(model, test_loader, device, patient):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct_predictions += preds.eq(targets).sum().item()
            total_samples += inputs.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct_predictions / total_samples * 100

    print(f'Test set: Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)')

    conf_matrix = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Confusion Matrix Test')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_all_patients/matrice_confusion_test.png')
    plt.close()


def plot_training_history(history, patient):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'history/training_history_{patient}.png')
    plt.show()


def compute_weight(y_train, y_val):
    train_zeros = np.sum(y_train == 0)
    train_ones = np.sum(y_train == 1)
    val_zeros = np.sum(y_val == 0)
    val_ones = np.sum(y_val == 1)

    total_zeros = train_zeros + val_zeros
    total_ones = train_ones + val_ones
    total = total_zeros + total_ones

    return total, total_zeros, total_ones


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean', device=torch.device('cpu')):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, outputs, targets):
        targets = targets.long().to(self.device)
        num_classes = outputs.shape[1]
        one_hot_targets = nn.functional.one_hot(targets, num_classes=num_classes).float().to(device=self.device)

        ce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, one_hot_targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        targets = targets.expand_as(outputs)
        return torch.mean(self.weights * (outputs - targets) ** 2)


def main():
    use_gpu = True
    if not torch.cuda.is_available():
        print('Not connected to a GPU')
    else:
        print('Connected to a GPU')

    torch.cuda.empty_cache()
    device = torch.device("cuda" if use_gpu else "cpu")

    file_path = '../data/patients/'

    patient_array = np.zeros((40000, 23, 2560), dtype=np.float32)
    label_array = np.zeros((40000,), dtype=np.int8)
    patients_list = ["under_sample_chb01", "under_sample_chb02", "under_sample_chb03",
                     "under_sample_chb05", "under_sample_chb06", "under_sample_chb07",
                     "under_sample_chb08", "under_sample_chb23", "under_sample_chb24"]
    index = 0
    memory_limit = 12000
    for patient in patients_list:
        check_memory_usage(memory_limit)

        data_patient, labels_patient = load_data(os.path.join(file_path, patient))
        num_samples = data_patient.shape[0]
        check_memory_usage(memory_limit)
        patient_array[index:index + num_samples] = data_patient
        label_array[index:index + num_samples] = labels_patient
        index += num_samples

    patients_array = patient_array[:index]
    labels_array = label_array[:index]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(patients_array, labels_array)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)

    num_epochs = 5
    model = Conformer().to(device)

    total, total_zeros, total_ones = compute_weight(y_train, y_val)
    weights = torch.tensor([total_ones/total, total_zeros/total], dtype=torch.float32).to(device)
    # Define the loss function with weights
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = WeightedMSELoss(weight=weights)
    criterion = FocalLoss(alpha=weights, gamma=2.0, device=device)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    model, history = train_model(model, train_loader, val_loader, device, criterion, num_epochs=num_epochs,
                                 patience=10, patient=patient)

    plot_training_history(history, patient)

    test_model(model, test_loader, device, patient)
    exit()


if __name__ == "__main__":
    main()
