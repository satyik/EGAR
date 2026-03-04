import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from train_scale import create_scale_c3

def get_dataloaders(batch_size=128):
    # Standard CIFAR-100 
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

def train_and_eval(epochs=100, batch_size=128, lr=0.001):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Advanced Training: SCALE-C3 on {device} ---")
    
    try:
        from train_scale import create_scale_c3
        model = create_scale_c3()
    except ImportError:
         print("Failed to import create_scale_c3. Make sure train_scale.py is correctly configured.")
         return
    
    # 1. Parameter Validation for SCALE-C3
    expected = 292960
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (Budget Hit: ~290k)")
    assert total_params == expected, f"Param mismatch! {total_params} != {expected}"
    
    model = model.to(device)
    
    # 2. Label Smoothing Regularization (Reduces Overconfidence on CIFAR-100)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 3. Base Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4) # Slightly higher weight decay 
    
    # 4. Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    trainloader, testloader = get_dataloaders(batch_size=batch_size)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 5. Gradient Clipping (Stabilizes deep recursive architectures)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}/{epochs}, Batch {i + 1:3d}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
        # Step the scheduler AFTER the epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
                
        # Evaluate
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
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
        print(f"Epoch {epoch + 1}/{epochs} (LR={current_lr:.5f}) | Test Accuracy -> Top-1: {acc_top1:.2f}% | Top-5: {acc_top5:.2f}%")
        
        # Save best model
        if acc_top1 > best_acc:
            print(f"[*] New best Top-1 Accuracy: {acc_top1:.2f}% (Previous: {best_acc:.2f}%). Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': acc_top1,
            }, "best_model_SCALE_C3_ADVANCED.pth")
            best_acc = acc_top1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Training Pipeline for SCALE-C3")
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs (Cosine Annealing tracks this)')
    parser.add_argument('--batch_size', type=int, default=128, help='Dataloader batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    args = parser.parse_args()

    train_and_eval(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
