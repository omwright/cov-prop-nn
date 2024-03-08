import os, sys, time
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

class CNet(nn.Module):
    """All-convolutional ReLU network based loosely on LeNet-5"""
    def __init__(self, input_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2), # (32, 28, 28)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=2), # (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=5, padding=2) # (64, 14, 14)
        )
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, stride=2), # (64, 7, 7)
            nn.Flatten(),
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes)
        ) # Partitioned for easy PL-DNN
    def forward(self, x):
        out = self.classifier(self.relu(self.features(x)))
        return out
    
def train_epoch(model, device, train_loader, criterion, optimizer):
    """Train over a single epoch"""
    model.train()
    avg_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(input)               # Forward pass
        loss = criterion(logits, target)    # Compute loss
        loss.backward()                     # Backpropagate
        optimizer.step()                    # Update weights
        avg_loss += loss.item()
    avg_loss /= len(train_loader) # Average batch loss
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            logits = model(input)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += loss.item()
    test_loss /= len(test_loader)
    accuracy = 100.*correct/len(test_loader.dataset)
    return test_loss, accuracy

if __name__ == "__main__":
    # Configure
    batch_size = 64
    learning_rate = 3e-4
    n_classes = 10
    n_epochs = 10
    fpath = "./data/"
    
    # Assign seed
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if cuda is available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0
    print(f"cuda = {cuda} with num_workers = {num_workers}, system version = {sys.version}")

    # Generate data
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST(root=fpath, 
                               train=True,
                               download=True,
                               transform=transform)
    test_set = datasets.MNIST(root=fpath, 
                              train=False, 
                              download=True,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    
    # Define model
    model = CNet(1, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3333, total_iters=n_epochs)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Train model
    best_loss = float('inf')
    print("Start training...")
    for epoch in range(1, n_epochs+1):
        start = time.time()
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, acc = test(model, device, test_loader)
        scheduler.step()
        end = time.time()
        if test_loss < best_loss:
            torch.save(model.state_dict(), fpath+"cnet.pt")
        print(f"Epoch {epoch}")
        print(f"\tRuntime: {(end - start):4f}\tTrain loss: {train_loss:4f}\tTest loss: {test_loss:4f}\tTest accuracy: {acc}")