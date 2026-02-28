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

def create_eg_opt_1_model():
    """
    Creates the EG-OPT-1 model:
    Strategy: Entropy Threshold and Dilation Tuning — Receptive Field Matched
    Channels (C): [48, 128, 256, 384]
    Recursion Depth (T): [3, 8, 8, 10]
    Dilations: [1, 2, 4]
    Expected Total Parameters: 175,696
    """
    return _BaseRecursiveArchitecture(
        Cs=[48, 128, 256, 384],
        Ts=[3, 8, 8, 10], 
        dilations=[1, 2, 4],
        num_classes=100
    )

def train_and_eval(epochs=10):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Training Candidate EG-OPT-1 on {device} ---")
    
    model = create_eg_opt_1_model()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params} (Expected: 175696)")
    assert total_params == 175696, "Parameter count mismatch for EG-OPT-1!"
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    trainloader, testloader = get_dataloaders()
    
    best_acc = 0.0
    
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
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                
                # Top 1 Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
                
                # Top 5 Accuracy
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
                correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                
        acc_top1 = 100 * correct_top1 / total
        acc_top5 = 100 * correct_top5 / total
        print(f"Epoch {epoch + 1} Test Accuracy -> Top-1: {acc_top1:.2f}% | Top-5: {acc_top5:.2f}%")
        
        # Save best model
        if acc_top1 > best_acc:
            print(f"[*] New best Top-1 Accuracy: {acc_top1:.2f}%. Saving model...")
            torch.save(model.state_dict(), f"best_model_opt_1.pth")
            best_acc = acc_top1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EG-OPT-1 Recursive Model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the EG-OPT-1 candidate')
    args = parser.parse_args()

    train_and_eval(epochs=args.epochs)
