import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from models import LSTMModel, GRUModel
from dataset import HARDataset, load_data
from config import class_mapping, HIDDEN_SIZE, BATCH_SIZE

def evaluate_model(model, test_loader, class_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_mapping.keys(),
                yticklabels=class_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_mapping.keys()))

def main():
    # Load data
    _, test_dataset, class_mapping = load_data()
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get input size and number of classes
    input_size = test_dataset[0][0].shape[0]  # number of features
    num_classes = len(class_mapping)

    # Eval LSTM Model    
    lstm_model = LSTMModel(input_size, HIDDEN_SIZE, num_classes)
    lstm_model.load_state_dict(torch.load('lstm_model.pth'))
    print("Evaluating LSTM model:")
    evaluate_model(lstm_model, test_loader, class_mapping)
    
    # Eval GRU Model
    gru_model = GRUModel(input_size, HIDDEN_SIZE, num_classes)
    gru_model.load_state_dict(torch.load('gru_model.pth'))
    print("\nEvaluating GRU model:")
    evaluate_model(gru_model, test_loader, class_mapping)

if __name__ == "__main__":
    main()