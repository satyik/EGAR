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
    # Parameter-free initial embedding: pad RGB (3 channels) to target_C with zeros
    B, C, H, W = x.shape
    padding = target_C - C
    x_padded = F.pad(x, (0, 0, 0, 0, 0, padding))
    return x_padded

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
                    padding = target_C - C
                    x = F.pad(x, (0, 0, 0, 0, 0, padding))
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
            
    test_params()
