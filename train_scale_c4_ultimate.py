import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from models_cifar import create_scale_c4_super

# CIFAR-100 Superclass mapping 
# Each index (0-99) maps to a superclass (0-19)
cifar100_to_superclass = torch.tensor([
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
    16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
    10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
    18, 1,  2, 15,  6,  0, 17,  8, 14, 13
])

def rand_bbox(size, lam):
    """Generates a random bounding box coordinates based on the lambda value."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(inputs, targets, alpha=1.0):
    """CutMix spatial blending"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    targets_a = targets
    targets_b = targets[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    
    # Clone before cutting
    inputs = inputs.clone()
    inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    
    return inputs, targets_a, targets_b, lam

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

def train_and_eval(epochs=150, batch_size=128, lr=0.001, cutmix_prob=0.5):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Ultimate Training: SCALE-C4 (CutMix + Superclass) on {device} ---")
    model = create_scale_c4_super(num_macro_classes=None).to(device)
    
    # 1. Parameter Validation
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (Dual Head Active)")
    
    # 2. Dual Loss Criterions
    criterion_fine = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_super = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    trainloader, testloader = get_dataloaders(batch_size=batch_size)
    superclass_map = cifar100_to_superclass.to(device)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels_fine = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            # Map Fine labels to Superclass exactly
            labels_super = superclass_map[labels_fine]
            
            optimizer.zero_grad()
            
            # 3. Apply CutMix Probabilistically
            r = np.random.rand(1)
            if r < cutmix_prob:
                inputs, targets_fine_a, targets_fine_b, lam = cutmix_data(inputs, labels_fine)
                
                # We also need to map the B targets logic to superclasses
                targets_super_a = superclass_map[targets_fine_a]
                targets_super_b = superclass_map[targets_fine_b]
                
                out_fine, out_super = model(inputs)
                
                # Calculate CutMix Loss for Fine
                loss_fine = criterion_fine(out_fine, targets_fine_a) * lam + criterion_fine(out_fine, targets_fine_b) * (1. - lam)
                
                # Calculate CutMix Loss for Superclass
                loss_super = criterion_super(out_super, targets_super_a) * lam + criterion_super(out_super, targets_super_b) * (1. - lam)
                
            else:
                out_fine, out_super = model(inputs)
                loss_fine = criterion_fine(out_fine, labels_fine)
                loss_super = criterion_super(out_super, labels_super)
            
            # 4. Superclass Hierarchical Loss (0.5 weight so it guides but doesn't overpower fine classification)
            loss = loss_fine + (0.5 * loss_super)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}/{epochs}, Batch {i + 1:3d}] Blended Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
                
        # Evaluate
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        correct_super = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                inputs, labels_fine = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                labels_super_target = superclass_map[labels_fine]
                
                out_fine, out_super = model(inputs)
                
                # Top 1 Accuracy Fine
                _, predicted = torch.max(out_fine.data, 1)
                total += labels_fine.size(0)
                correct_top1 += (predicted == labels_fine).sum().item()
                
                # Top 1 Accuracy Superclass
                _, predicted_sup = torch.max(out_super.data, 1)
                correct_super += (predicted_sup == labels_super_target).sum().item()
                
                # Top 5 Accuracy Fine
                _, pred_top5 = out_fine.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct = pred_top5.eq(labels_fine.view(1, -1).expand_as(pred_top5))
                correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                
        acc_top1 = 100 * correct_top1 / total
        acc_top5 = 100 * correct_top5 / total
        acc_super = 100 * correct_super / total
        
        print(f"Epoch {epoch + 1}/{epochs} (LR={current_lr:.5f})")
        print(f"Test Accuracy -> Superclass: {acc_super:.2f}% | Fine Top-1: {acc_top1:.2f}% | Fine Top-5: {acc_top5:.2f}%")
        
        if acc_top1 > best_acc:
            print(f"[*] New best Top-1 Fine Accuracy: {acc_top1:.2f}%. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': acc_top1,
            }, "best_model_SCALE_C4_ULTIMATE.pth")
            best_acc = acc_top1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Training Pipeline: SCALE-C4")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability to apply CutMix augmentation')
    args = parser.parse_args()

    train_and_eval(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, cutmix_prob=args.cutmix_prob)
