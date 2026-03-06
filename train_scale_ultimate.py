import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from models_cifar import create_scale_c3_super

# CIFAR-100 Superclass mapping 
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

cifar100_super_to_macro = torch.tensor([
    3, 3, 0, 1, 0, 1, 1, 4, 5, 2, 2, 5, 6, 4, 6, 7, 6, 0, 2, 2
])

# CIFAR-10 Superclass Mapping (Animals vs Vehicles)
cifar10_to_superclass = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]) # 0: Animal, 1: Vehicle
cifar10_super_to_macro = torch.tensor([0, 0]) # everything to 0 for a dummy 3rd head on CIFAR-10

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w"
        self.base_optimizer.step()  # do the actual "w" update
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(inputs, targets, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    targets_a = targets
    targets_b = targets[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs = inputs.clone()
    inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    return inputs, targets_a, targets_b, lam

def get_dataloaders(dataset_name="cifar100", batch_size=128):
    # 1. Added RandAugment for complex geometric/photometric distortions
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9), 
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if dataset_name == "cifar100" else transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if dataset_name == "cifar100" else transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    if dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        # Placeholder for Tiny-ImageNet and iNaturalist (requires custom dataset implementations)
        raise NotImplementedError(f"Dataset {dataset_name} is not fully integrated in standard script yet.")
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def train_and_eval(epochs=150, batch_size=128, lr=0.001, cutmix_prob=0.5, dataset="cifar100"):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Ultimate Training: SCALE-C3 (SAM + CutMix + RandAugment + Hierarchy) on {device} ---")
    
    if dataset == "cifar100":
        model = create_scale_c3_super(num_classes=100, num_super_classes=20, num_macro_classes=8).to(device)
        superclass_map = cifar100_to_superclass.to(device)
        macro_map = cifar100_super_to_macro.to(device)
    elif dataset == "cifar10":
        model = create_scale_c3_super(num_classes=10, num_super_classes=2, num_macro_classes=1).to(device)
        superclass_map = cifar10_to_superclass.to(device)
        macro_map = cifar10_super_to_macro.to(device)
    else:
        raise NotImplementedError()
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (Triple Head Active)")
    
    criterion_fine = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_super = nn.CrossEntropyLoss()
    criterion_macro = nn.CrossEntropyLoss()
    
    # 2. Replaced AdamW with SAM wrapper utilizing AdamW as the base
    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=lr, weight_decay=5e-4) 
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs, eta_min=1e-5)
    
    trainloader, testloader = get_dataloaders(dataset_name=dataset, batch_size=batch_size)
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels_fine = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            labels_super = superclass_map[labels_fine]
            labels_macro = macro_map[labels_super]
            
            optimizer.zero_grad()
            
            r = np.random.rand(1)
            if r < cutmix_prob:
                inputs_cut, targets_fine_a, targets_fine_b, lam = cutmix_data(inputs, labels_fine)
                targets_super_a = superclass_map[targets_fine_a]
                targets_super_b = superclass_map[targets_fine_b]
                targets_macro_a = macro_map[targets_super_a]
                targets_macro_b = macro_map[targets_super_b]
                
                # SAM Pass 1: Climb to the steepest/sharpest local gradient
                out_fine, out_super, out_macro = model(inputs_cut)
                loss_fine = criterion_fine(out_fine, targets_fine_a) * lam + criterion_fine(out_fine, targets_fine_b) * (1. - lam)
                loss_super = criterion_super(out_super, targets_super_a) * lam + criterion_super(out_super, targets_super_b) * (1. - lam)
                loss_macro = criterion_macro(out_macro, targets_macro_a) * lam + criterion_macro(out_macro, targets_macro_b) * (1. - lam)
                loss = loss_fine + (0.5 * loss_super) + (0.2 * loss_macro)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)
                
                # SAM Pass 2: At that sharpest point, calculate the real flat-valley update
                out_fine, out_super, out_macro = model(inputs_cut)
                loss_fine = criterion_fine(out_fine, targets_fine_a) * lam + criterion_fine(out_fine, targets_fine_b) * (1. - lam)
                loss_super = criterion_super(out_super, targets_super_a) * lam + criterion_super(out_super, targets_super_b) * (1. - lam)
                loss_macro = criterion_macro(out_macro, targets_macro_a) * lam + criterion_macro(out_macro, targets_macro_b) * (1. - lam)
                loss = loss_fine + (0.5 * loss_super) + (0.2 * loss_macro)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
                
            else:
                # SAM Pass 1
                out_fine, out_super, out_macro = model(inputs)
                loss_fine = criterion_fine(out_fine, labels_fine)
                loss_super = criterion_super(out_super, labels_super)
                loss_macro = criterion_macro(out_macro, labels_macro)
                loss = loss_fine + (0.5 * loss_super) + (0.2 * loss_macro)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)
                
                # SAM Pass 2
                out_fine, out_super, out_macro = model(inputs)
                loss_fine = criterion_fine(out_fine, labels_fine)
                loss_super = criterion_super(out_super, labels_super)
                loss_macro = criterion_macro(out_macro, labels_macro)
                loss = loss_fine + (0.5 * loss_super) + (0.2 * loss_macro)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}/{epochs}, Batch {i + 1:3d}] Blended SAM Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
                
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        correct_super = 0
        correct_macro = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                inputs, labels_fine = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                labels_super_target = superclass_map[labels_fine]
                labels_macro_target = macro_map[labels_super_target]
                
                out_fine, out_super, out_macro = model(inputs)
                
                _, predicted = torch.max(out_fine.data, 1)
                total += labels_fine.size(0)
                correct_top1 += (predicted == labels_fine).sum().item()
                
                _, predicted_sup = torch.max(out_super.data, 1)
                correct_super += (predicted_sup == labels_super_target).sum().item()
                
                _, predicted_mac = torch.max(out_macro.data, 1)
                correct_macro += (predicted_mac == labels_macro_target).sum().item()
                
                pred_top5 = out_fine.topk(5, 1, True, True)[1].t() if out_fine.shape[1] >= 5 else out_fine.topk(1, 1, True, True)[1].t()
                correct = pred_top5.eq(labels_fine.view(1, -1).expand_as(pred_top5))
                correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                
        acc_top1 = 100 * correct_top1 / total
        acc_top5 = 100 * (correct_top5 / total) if out_fine.shape[1] >= 5 else acc_top1
        acc_super = 100 * correct_super / total
        acc_macro = 100 * correct_macro / total
        print(f"Epoch {epoch + 1}/{epochs} (LR={current_lr:.5f}) | SAM Optimization Frame")
        print(f"Test Accuracy -> Macroclass: {acc_macro:.2f}% | Superclass: {acc_super:.2f}% | Fine Top-1: {acc_top1:.2f}% | Fine Top-5: {acc_top5:.2f}%")
        
        if acc_top1 > best_acc:
            print(f"[*] New best Top-1 Fine Accuracy: {acc_top1:.2f}%. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': acc_top1,
            }, "best_model_SCALE_C3_ULTIMATE_SAM.pth")
            best_acc = acc_top1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Training Pipeline: SCALE-C3")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability to apply CutMix augmentation')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tiny_imagenet', 'inaturalist'])
    args = parser.parse_args()

    train_and_eval(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, cutmix_prob=args.cutmix_prob, dataset=args.dataset)
