import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import LSTMModel, GRUModel
from dataset import HARDataset, load_data
from config import class_mapping, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0    

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return model

def main():
    # Load data
    train_dataset, test_dataset, class_mapping = load_data()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get input size and number of classes
    input_size = train_dataset[0][0].shape[0]  # number of features
    num_classes = len(class_mapping)
    
    # Initialize models
    lstm_model = LSTMModel(input_size, HIDDEN_SIZE, num_classes)
    gru_model = GRUModel(input_size, HIDDEN_SIZE, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    
    # Train and save LSTM model
    print("Training LSTM model...")
    trained_lstm = train_model(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, NUM_EPOCHS)
    torch.save(trained_lstm.state_dict(), 'lstm_model.pth')
    
    # Train and save GRU model
    print("Training GRU model...")
    trained_gru = train_model(gru_model, train_loader, val_loader, criterion, gru_optimizer, NUM_EPOCHS)
    torch.save(trained_gru.state_dict(), 'gru_model.pth')
    

if __name__ == "__main__":
    main()