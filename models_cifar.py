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
        
    def compute_entropy_gate(self, x):
        # Entropy-gated residual path computation (Parameter-free)
        gates = []
        for d in self.dilations:
            shifted_h = torch.roll(x, shifts=d, dims=2)
            shifted_w = torch.roll(x, shifts=d, dims=3)
            diff = torch.abs(x - shifted_h) + torch.abs(x - shifted_w)
            gates.append(diff)
        
        # Mean variance-like representation across dilations
        gate = torch.mean(torch.stack(gates, dim=0), dim=0)
        # Bounded to [0, 1] using Sigmoid
        gate = torch.sigmoid(gate)
        return gate

    def forward(self, x):
        for t in range(self.T):
            # Dynamic entropy gate based on the current feature map structure
            gate = self.compute_entropy_gate(x)
            
            # Application of W_shared
            h = self.conv1x1(x)
            
            # ShuffleNet channel mix for groups=2
            B_h, C_h, H_h, W_h = h.shape
            h = h.view(B_h, 2, C_h // 2, H_h, W_h).transpose(1, 2).reshape(B_h, C_h, H_h, W_h)
            
            h = self.conv3x3_dw(h)
            
            # IS-Norm (Instance Normalization without params + manual Iteration-Specific gamma/beta)
            h_normed = F.instance_norm(h)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            
            h_act = F.gelu(h_normed)
            
            # Entropy-gated residual update
            x = x + h_act * gate
            
        return x

# The parameter-free handling is now dynamically done inside the main loop based on downsampling schedules.

def parameter_free_stem(x, target_C):
    # Parameter-free initial embedding: tile RGB channels to target_C 
    B, C, H, W = x.shape
    repeats = (target_C + C - 1) // C
    x_tiled = x.repeat(1, repeats, 1, 1)
    return x_tiled[:, :target_C, :, :]

class _BaseRecursiveArchitecture(nn.Module):
    def __init__(self, Cs, Ts, downsamples=None, dilations=[1, 2, 4], num_classes=100):
        super().__init__()
        self.Cs = Cs
        if downsamples is None:
            # By default, downsample at every stage except the first one
            downsamples = [False] + [True] * (len(Cs) - 1)
        self.downsamples = downsamples
        
        self.stages = nn.ModuleList()
        for C, T in zip(Cs, Ts):
            self.stages.append(RecursiveBlock(C, T, dilations=dilations))
            
        # Classifier Head (C_final * 100 parameters)
        self.classifier = nn.Linear(Cs[-1], num_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                # Parameter-free channel adjustment
                B, C, H, W = x.shape
                target_C = self.Cs[i]
                if target_C > C:
                    repeats = (target_C + C - 1) // C
                    x = x.repeat(1, repeats, 1, 1)[:, :target_C, :, :]
                elif target_C < C:
                    x = x[:, :target_C, :, :]
                    
            x = stage(x)
            
        # Spatial pooling mapping spatial resolution to structural vector
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(candidate_id):
    if candidate_id == 1:
        # Candidate 1: Deep & Narrow
        return _BaseRecursiveArchitecture(Cs=[32, 64, 128, 256], Ts=[2, 4, 8, 10])
    elif candidate_id == 2:
        # Candidate 2: Wide & Shallow
        return _BaseRecursiveArchitecture(Cs=[64, 96, 144, 216], Ts=[2, 2, 4, 4])
    elif candidate_id == 3:
        # Candidate 3: Funnel
        return _BaseRecursiveArchitecture(Cs=[48, 96, 192, 384], Ts=[2, 2, 4, 10])
    else:
        raise ValueError("Invalid candidate_id")

class RecursiveBlockPlus(nn.Module):
    def __init__(self, C, T, dilations=[1, 2, 4], se_ratio=16):
        super().__init__()
        self.C = C
        self.T = T
        self.dilations = dilations
        
        # 1x1 dense conv (groups=1), C^2 parameters
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1, bias=False)
        
        # 7x7 depthwise conv, 49C parameters
        self.conv7x7_dw = nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C, bias=False)
        
        # Squeeze-and-Excitation (SE) Block
        self.se_fc1 = nn.Linear(C, C // se_ratio, bias=False)
        self.se_fc2 = nn.Linear(C // se_ratio, C, bias=False)
        
        # IS-Norm Params (2 * T * C)
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas = nn.Parameter(torch.zeros(T, 1, C, 1, 1))
        
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
        for t in range(self.T):
            gate = self.compute_entropy_gate(x)
            
            h = self.conv1x1(x)
            h = self.conv7x7_dw(h)
            
            h_normed = F.instance_norm(h)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            h_act = F.gelu(h_normed)
            
            # Squeeze-and-Excitation computation
            B, C_h, _, _ = h_act.shape
            se = F.adaptive_avg_pool2d(h_act, 1).view(B, C_h)
            se = F.relu(self.se_fc1(se))
            se = torch.sigmoid(self.se_fc2(se)).view(B, C_h, 1, 1)
            
            # Gated SE Residual update
            x = x + (h_act * se * gate)
            
        return x

class _ScaleC3PlusArchitecture(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.Cs = [64, 256, 224, 256, 384, 320]
        self.downsamples = [False, True, True, False, True, False]
        self.Ts = [8, 10, 10, 10, 10, 10]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        for C, T in zip(self.Cs, self.Ts):
            self.stages.append(RecursiveBlockPlus(C, T, dilations=self.dilations))
            
        # Classifier Head (C_final * 100 parameters)
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                # Parameter-free channel adjustment
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
        x = self.classifier(x)
        return x

def create_scale_c3_plus():
    return _ScaleC3PlusArchitecture(num_classes=100)

class RecursiveBlockSym(nn.Module):
    def __init__(self, C, T, dilations=[1, 2, 4], se_ratio=16, drop_prob=0.2):
        super().__init__()
        self.C = C
        self.T = T
        self.dilations = dilations
        self.drop_prob = drop_prob
        
        # 1x1 dense conv (groups=1), C^2 parameters
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1, bias=False)
        
        # Split-Channels Multi-Branch
        # branch 1: C/4 (1x1)
        # branch 2: C/4 (3x3 dw)
        # branch 3: C/2 (7x7 dw)
        self.c1 = C // 4
        self.c2 = C // 4
        self.c3 = C - self.c1 - self.c2 # To handle any rounding issues precisely
        
        self.branch1 = nn.Conv2d(self.c1, self.c1, kernel_size=1, bias=False)
        self.branch2 = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, groups=self.c2, bias=False)
        self.branch3 = nn.Conv2d(self.c3, self.c3, kernel_size=7, padding=3, groups=self.c3, bias=False)
        
        # Squeeze-and-Excitation (SE) Block on concatenated features (which sum back to C)
        self.se_fc1 = nn.Linear(C, C // se_ratio, bias=False)
        self.se_fc2 = nn.Linear(C // se_ratio, C, bias=False)
        
        # IS-Norm Params (2 * T * C)
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas = nn.Parameter(torch.zeros(T, 1, C, 1, 1))
        
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
        for t in range(self.T):
            # Stochastic Depth
            if self.training and torch.rand(1).item() < self.drop_prob:
                continue 
                
            gate = self.compute_entropy_gate(x)
            
            h = self.conv1x1(x)
            
            # Split features along channel dimension
            h1, h2, h3 = torch.split(h, [self.c1, self.c2, self.c3], dim=1)
            
            b1 = self.branch1(h1)
            b2 = self.branch2(h2)
            b3 = self.branch3(h3)
            
            h_concat = torch.cat([b1, b2, b3], dim=1)
            
            h_normed = F.instance_norm(h_concat)
            h_normed = h_normed * self.gammas[t] + self.betas[t]
            h_act = F.gelu(h_normed)
            
            # Squeeze-and-Excitation computation
            B, C_h, _, _ = h_act.shape
            se = F.adaptive_avg_pool2d(h_act, 1).view(B, C_h)
            se = F.relu(self.se_fc1(se))
            se = torch.sigmoid(self.se_fc2(se)).view(B, C_h, 1, 1)
            
            h_excited = h_act * se
            
            # Gated SE Residual update
            x = x + (h_excited * gate)
            
        return x

class _ScaleC4Architecture(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.Cs = [64, 128, 256, 512]
        self.downsamples = [False, True, True, True]
        self.Ts = [8, 10, 10, 10]
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        # Drop paths for stochastic depth: [0.0, 0.1, 0.2, 0.3]
        drop_probs = [0.0, 0.1, 0.2, 0.3]
        for C, T, dp in zip(self.Cs, self.Ts, drop_probs):
            self.stages.append(RecursiveBlockSym(C, T, dilations=self.dilations, drop_prob=dp))
            
        # Classifier Head (C_final * 100 parameters)
        self.classifier = nn.Linear(self.Cs[-1], num_classes, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.downsamples[i]:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                
                # Parameter-free channel adjustment (Tiling)
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
        x = self.classifier(x)
        return x

def create_scale_c4():
    return _ScaleC4Architecture(num_classes=100)

if __name__ == "__main__":
    def test_params():
        expected_params = {
            1: 81248,
            2: 70152,
            3: 152592
        }
        for i in [1, 2, 3]:
            model = create_model(i)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Candidate {i} - Total Params: {total_params} (Expected: {expected_params[i]})")
            assert total_params == expected_params[i], f"Param mismatch exactly for Candidate {i}"
            
        # SCALE-C3+ Math Check
        model_plus = create_scale_c3_plus()
        total_plus = sum(p.numel() for p in model_plus.parameters() if p.requires_grad)
        print(f"SCALE-C3+ Total Params: {total_plus}")
            
        # SCALE-C4 Math Check
        model_c4 = create_scale_c4()
        total_c4 = sum(p.numel() for p in model_c4.parameters() if p.requires_grad)
        print(f"SCALE-C4 Total Params: {total_c4}")
            
    test_params()
