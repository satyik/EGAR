import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import argparse

from models_cifar import _BaseRecursiveArchitecture

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    return trainloader, testloader

def create_mutation_a():
    """
    EG-OPT-1-MA
    Strategy: IS-Norm Collapse Fix — Dual 8x8 Sub-Stage + Stage 4 Compression
    Stages:
      1: 32x32 (C=48, T=2)
      2: 16x16 (C=128, T=10)
      3a: 8x8 (C=192, T=6)
      3b: 8x8 (C=224, T=10)
      4: 4x4 (C=320, T=10)
    """
    return _BaseRecursiveArchitecture(
        Cs=[48, 128, 192, 224, 320],
        Ts=[2, 10, 6, 10, 10], 
        downsamples=[False, True, True, False, True], # False = Maintain resolution
        dilations=[1, 2, 4],
        num_classes=100
    )

def create_mutation_b():
    """
    Mutation B (EG-OPT-1-MB)
    Strategy: Recursive Stem Expansion + Aggressive Late Deepening
    Stages:
      1: 32x32 (C=64, T=6)
      2: 16x16 (C=192, T=10)
      3: 8x8 (C=192, T=6)
      4: 4x4 (C=288, T=8)
      5: 4x4 (C=256, T=8)
    """
    return _BaseRecursiveArchitecture(
        Cs=[64, 192, 192, 288, 256],
        Ts=[6, 10, 6, 8, 8],
        downsamples=[False, True, True, True, False],
        dilations=[1, 2, 4],
        num_classes=100
    )

def train_and_eval(model_name, epochs=10):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Training Candidate {model_name} on {device} ---")
    
    if model_name == "EG-OPT-1-MA":
        model = create_mutation_a()
        expected = 160208
    elif model_name == "EG-OPT-1-MB":
        model = create_mutation_b()
        expected = 163296
    else:
        raise ValueError("Invalid model name")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params} (Expected Dynamic Math: {expected})")
    assert total_params == expected, f"Param mismatch! {total_params} != {expected}"
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    trainloader, testloader = get_dataloaders()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1:3d}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch + 1} Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mutation Architectures")
    parser.add_argument('--model', type=str, choices=['MA', 'MB', 'BOTH'], default='BOTH', help='Which mutation to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    if args.model in ['MA', 'BOTH']:
        train_and_eval("EG-OPT-1-MA", epochs=args.epochs)
    if args.model in ['MB', 'BOTH']:
        train_and_eval("EG-OPT-1-MB", epochs=args.epochs)
