import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import get_model

# Config
PROCESSED_DIR = "data/processed"
EPOCHS = 50 # Start small to make sure it works
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def train():
    # 1. Load Data
    train_loader, val_loader, classes = get_loaders(PROCESSED_DIR, BATCH_SIZE)
    print(f"Loaded {len(classes)} classes: {classes}")

    # 2. Setup Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

    # 4. Save the "Neural" Brain
    torch.save(model.state_dict(), "brain_model_neural.pth")
    print("✅ Training complete! Model saved as brain_model_neural.pth")

if __name__ == "__main__":
    train()