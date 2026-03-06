import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveBlock(nn.Module):
    def __init__(self, C, T, dilations=[1, 2, 4]):
        super().__init__()
        self.C = C
        self.T = T
        self.dilations = dilations
        
        # 1x1 conv with groups=2, C^2/2 parameters
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1, groups=2, bias=False)
        
        # 3x3 depthwise conv, 9C parameters
        self.conv3x3_dw = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        
        # IS-Norm Params (2 * T * C)
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas = nn.Parameter(torch.zeros(T, 1, C, 1, 1))
        
        # Temporal Feature Mixing momentum
        self.alpha = nn.Parameter(torch.tensor(0.25))
        
    def compute_entropy_gate(self, x):
        gates = []
        for d in self.dilations:
            shifted_h = torch.roll(x, shifts=d, dims=2)
            shifted_w = torch.roll(x, shifts=d, dims=3)
            diff = torch.abs(x - shifted_h) + torch.abs(x - shifted_w)
            gates.append(diff)
        
        gate = torch.mean(torch.stack(gates, dim=0), dim=0)
        gate = torch.sigmoid(gate)
        return gate

    def forward(self, x):
        x_prev = x
        for t in range(self.T):
            gate = self.compute_entropy_gate(x)
            
            h = self.conv1x1(x)
            
            B_h, C_h, H_h, W_h = h.shape
            h = h.view(B_h, 2, C_h // 2, H_h, W_h).transpose(1, 2).reshape(B_h, C_h, H_h, W_h)
            
            h = self.conv3x3_dw(h)
            
            h_normed = F.instance_norm(h)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            
            h_act = F.gelu(h_normed)
            
            x_new = x + h_act * gate + self.alpha * (x - x_prev)
            x_prev = x
            x = x_new
            
        return x

def parameter_free_stem(x, target_C):
    B, C, H, W = x.shape
    repeats = (target_C + C - 1) // C
    x_tiled = x.repeat(1, repeats, 1, 1)
    return x_tiled[:, :target_C, :, :]

class _BaseRecursiveArchitecture(nn.Module):
    def __init__(self, Cs, Ts, downsamples=None, dilations=[1, 2, 4], num_classes=100, num_super_classes=None, num_macro_classes=None):
        super().__init__()
        self.Cs = Cs
        if downsamples is None:
            downsamples = [False] + [True] * (len(Cs) - 1)
        self.downsamples = downsamples
        
        self.stages = nn.ModuleList()
        for C, T in zip(Cs, Ts):
            self.stages.append(RecursiveBlock(C, T, dilations=dilations))
            
        self.classifier = nn.Linear(Cs[-1], num_classes, bias=False)
        self.use_super = num_super_classes is not None
        if self.use_super:
            self.classifier_super = nn.Linear(Cs[-1], num_super_classes, bias=False)
        self.use_macro = num_macro_classes is not None
        if self.use_macro:
            self.classifier_macro = nn.Linear(Cs[-1], num_macro_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        out = self.classifier(x)
        if self.use_super and self.use_macro:
            return out, self.classifier_super(x), self.classifier_macro(x)
        elif self.use_super:
            return out, self.classifier_super(x)
        return out

class _ScaleC3SuperArchitecture(nn.Module):
    def __init__(self, num_classes=100, num_super_classes=20, num_macro_classes=8):
        super().__init__()
        self.Cs = [64, 256, 224, 256, 384, 320]
        self.Ts = [8, 10, 10, 10, 10, 10]
        self.downsamples = [False, True, True, False, True, False]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        for C, T in zip(self.Cs, self.Ts):
            self.stages.append(RecursiveBlock(C, T, dilations=self.dilations))
            
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        self.use_super = num_super_classes is not None
        if self.use_super:
            self.classifier_super = nn.Linear(self.Cs[-1], num_super_classes, bias=False)
        self.use_macro = num_macro_classes is not None
        if self.use_macro:
            self.classifier_macro = nn.Linear(self.Cs[-1], num_macro_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        out = self.classifier(x)
        if self.use_super and self.use_macro:
            return out, self.classifier_super(x), self.classifier_macro(x)
        elif self.use_super:
            return out, self.classifier_super(x)
        return out

def create_scale_c3_super(num_classes=100, num_super_classes=20, num_macro_classes=8):
    return _ScaleC3SuperArchitecture(num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)

def create_model(candidate_id, num_classes=100, num_super_classes=None, num_macro_classes=None):
    if candidate_id == 1:
        return _BaseRecursiveArchitecture(Cs=[32, 64, 128, 256], Ts=[2, 4, 8, 10], num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)
    elif candidate_id == 2:
        return _BaseRecursiveArchitecture(Cs=[64, 96, 144, 216], Ts=[2, 2, 4, 4], num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)
    elif candidate_id == 3:
        return _BaseRecursiveArchitecture(Cs=[48, 96, 192, 384], Ts=[2, 2, 4, 10], num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)
    else:
        raise ValueError("Invalid candidate_id")

class RecursiveBlockPlus(nn.Module):
    def __init__(self, C, T, dilations=[1, 2, 4], se_ratio=16):
        super().__init__()
        self.C = C
        self.T = T
        self.dilations = dilations
        
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.conv7x7_dw = nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C, bias=False)
        self.se_fc1 = nn.Linear(C, C // se_ratio, bias=False)
        self.se_fc2 = nn.Linear(C // se_ratio, C, bias=False)
        
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas = nn.Parameter(torch.zeros(T, 1, C, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.25))
        
    def compute_entropy_gate(self, x):
        gates = []
        for d in self.dilations:
            shifted_h = torch.roll(x, shifts=d, dims=2)
            shifted_w = torch.roll(x, shifts=d, dims=3)
            diff = torch.abs(x - shifted_h) + torch.abs(x - shifted_w)
            gates.append(diff)
        
        gate = torch.mean(torch.stack(gates, dim=0), dim=0)
        gate = torch.sigmoid(gate)
        return gate

    def forward(self, x):
        x_prev = x
        for t in range(self.T):
            gate = self.compute_entropy_gate(x)
            
            h = self.conv1x1(x)
            h = self.conv7x7_dw(h)
            
            h_normed = F.instance_norm(h)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            h_act = F.gelu(h_normed)
            
            B, C_h, _, _ = h_act.shape
            se = F.adaptive_avg_pool2d(h_act, 1).view(B, C_h)
            se = F.relu(self.se_fc1(se))
            se = torch.sigmoid(self.se_fc2(se)).view(B, C_h, 1, 1)
            
            x_new = x + (h_act * se * gate) + self.alpha * (x - x_prev)
            x_prev = x
            x = x_new
            
        return x

class _ScaleC3PlusArchitecture(nn.Module):
    def __init__(self, num_classes=100, num_super_classes=None, num_macro_classes=None):
        super().__init__()
        self.Cs = [64, 256, 224, 256, 384, 320]
        self.downsamples = [False, True, True, False, True, False]
        self.Ts = [8, 10, 10, 10, 10, 10]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        for C, T in zip(self.Cs, self.Ts):
            self.stages.append(RecursiveBlockPlus(C, T, dilations=self.dilations))
            
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        self.use_super = num_super_classes is not None
        if self.use_super:
            self.classifier_super = nn.Linear(self.Cs[-1], num_super_classes, bias=False)
        self.use_macro = num_macro_classes is not None
        if self.use_macro:
            self.classifier_macro = nn.Linear(self.Cs[-1], num_macro_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        out = self.classifier(x)
        if self.use_super and self.use_macro:
            return out, self.classifier_super(x), self.classifier_macro(x)
        elif self.use_super:
            return out, self.classifier_super(x)
        return out

def create_scale_c3_plus(num_classes=100, num_super_classes=None, num_macro_classes=None):
    return _ScaleC3PlusArchitecture(num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)

class RecursiveBlockSym(nn.Module):
    def __init__(self, C, T, dilations=[1, 2, 4], se_ratio=16, drop_prob=0.2):
        super().__init__()
        self.C = C
        self.T = T
        self.dilations = dilations
        self.drop_prob = drop_prob
        
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1, bias=False)
        
        self.c1 = C // 4
        self.c2 = C // 4
        self.c3 = C - self.c1 - self.c2
        
        self.branch1 = nn.Conv2d(self.c1, self.c1, kernel_size=1, bias=False)
        self.branch2 = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, groups=self.c2, bias=False)
        self.branch3 = nn.Conv2d(self.c3, self.c3, kernel_size=7, padding=3, groups=self.c3, bias=False)
        
        self.se_fc1 = nn.Linear(C, C // se_ratio, bias=False)
        self.se_fc2 = nn.Linear(C // se_ratio, C, bias=False)
        
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas = nn.Parameter(torch.zeros(T, 1, C, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.25))
        
    def compute_entropy_gate(self, x):
        gates = []
        for d in self.dilations:
            shifted_h = torch.roll(x, shifts=d, dims=2)
            shifted_w = torch.roll(x, shifts=d, dims=3)
            diff = torch.abs(x - shifted_h) + torch.abs(x - shifted_w)
            gates.append(diff)
        
        gate = torch.mean(torch.stack(gates, dim=0), dim=0)
        gate = torch.sigmoid(gate)
        return gate

    def forward(self, x):
        x_prev = x
        for t in range(self.T):
            if self.training and torch.rand(1).item() < self.drop_prob:
                continue 
                
            gate = self.compute_entropy_gate(x)
            
            h = self.conv1x1(x)
            
            h1, h2, h3 = torch.split(h, [self.c1, self.c2, self.c3], dim=1)
            
            b1 = self.branch1(h1)
            b2 = self.branch2(h2)
            b3 = self.branch3(h3)
            
            h_concat = torch.cat([b1, b2, b3], dim=1)
            
            h_normed = F.instance_norm(h_concat)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            h_act = F.gelu(h_normed)
            
            B, C_h, _, _ = h_act.shape
            se = F.adaptive_avg_pool2d(h_act, 1).view(B, C_h)
            se = F.relu(self.se_fc1(se))
            se = torch.sigmoid(self.se_fc2(se)).view(B, C_h, 1, 1)
            
            h_excited = h_act * se
            
            x_new = x + (h_excited * gate) + self.alpha * (x - x_prev)
            x_prev = x
            x = x_new
            
        return x

class _ScaleC4Architecture(nn.Module):
    def __init__(self, num_classes=100, num_super_classes=None, num_macro_classes=None):
        super().__init__()
        self.Cs = [64, 128, 256, 512]
        self.downsamples = [False, True, True, True]
        self.Ts = [8, 10, 10, 10]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        drop_probs = [0.0, 0.1, 0.2, 0.3]
        for C, T, dp in zip(self.Cs, self.Ts, drop_probs):
            self.stages.append(RecursiveBlockSym(C, T, dilations=self.dilations, drop_prob=dp))
            
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        self.use_super = num_super_classes is not None
        if self.use_super:
            self.classifier_super = nn.Linear(self.Cs[-1], num_super_classes, bias=False)
        self.use_macro = num_macro_classes is not None
        if self.use_macro:
            self.classifier_macro = nn.Linear(self.Cs[-1], num_macro_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        out = self.classifier(x)
        if self.use_super and self.use_macro:
            return out, self.classifier_super(x), self.classifier_macro(x)
        elif self.use_super:
            return out, self.classifier_super(x)
        return out

def create_scale_c4(num_classes=100, num_super_classes=None, num_macro_classes=None):
    return _ScaleC4Architecture(num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)

class _ScaleC4SuperArchitecture(nn.Module):
    def __init__(self, num_classes=100, num_super_classes=20, num_macro_classes=8):
        super().__init__()
        self.Cs = [64, 128, 256, 512]
        self.downsamples = [False, True, True, True]
        self.Ts = [8, 10, 10, 10]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        drop_probs = [0.0, 0.1, 0.2, 0.3]
        for C, T, dp in zip(self.Cs, self.Ts, drop_probs):
            self.stages.append(RecursiveBlockSym(C, T, dilations=self.dilations, drop_prob=dp))
            
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        self.use_super = num_super_classes is not None
        if self.use_super:
            self.classifier_super = nn.Linear(self.Cs[-1], num_super_classes, bias=False)
        self.use_macro = num_macro_classes is not None
        if self.use_macro:
            self.classifier_macro = nn.Linear(self.Cs[-1], num_macro_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        out = self.classifier(x)
        if self.use_super and self.use_macro:
            return out, self.classifier_super(x), self.classifier_macro(x)
        elif self.use_super:
            return out, self.classifier_super(x)
        return out

def create_scale_c4_super(num_classes=100, num_super_classes=20, num_macro_classes=8):
    return _ScaleC4SuperArchitecture(num_classes=num_classes, num_super_classes=num_super_classes, num_macro_classes=num_macro_classes)

if __name__ == "__main__":
    def test_params():
        pass
    test_params()
