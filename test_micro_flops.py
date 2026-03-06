import torch
import thop
import torch.nn as nn
import torch.nn.functional as F
from models_cifar import RecursiveBlock, parameter_free_stem

class _ScaleMicroSearch(nn.Module):
    def __init__(self, Cs, Ts, downsamples):
        super().__init__()
        self.Cs = Cs
        self.Ts = Ts
        self.downsamples = downsamples
        self.dilations = [1, 2, 4]
        
        self.stages = nn.ModuleList()
        for C, T in zip(self.Cs, self.Ts):
            self.stages.append(RecursiveBlock(C, T, dilations=self.dilations))
            
        self.classifier = nn.Linear(self.Cs[-1], 100, bias=False)
        self.classifier_super = nn.Linear(self.Cs[-1], 20, bias=False)
        
    def forward(self, x):
        x = parameter_free_stem(x, self.Cs[0])
        # Force stem spatial downsample if downsamples[0] is true
        if self.downsamples[0]:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            
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
        return self.classifier(x), self.classifier_super(x)

def test_config(Cs, Ts, downsamples, name):
    model = _ScaleMicroSearch(Cs, Ts, downsamples)
    input_tensor = torch.randn(1, 3, 32, 32)
    macs, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    flops = (macs * 2) / 1e9
    print(f"[{name}] Params: {params:,.0f} | GFLOPs: {flops:.4f}")
    return flops

if __name__ == "__main__":
    print("--- Searching for <0.1 GFLOPs ---")
    
    # 1. Start Stem at 16x16, limit early T
    test_config([64, 256, 224, 256, 384, 320], [1, 3, 11, 13, 16, 12], [True, True, True, False, True, False], "<0.1 GFLOP Attempt 1 (Shrinking early T)")
    
    # 2. Aggressive Stem at 8x8 immediately, stack everything to the deep end
    test_config([64, 256, 224, 256, 384, 320], [2, 4, 12, 14, 14, 10], [True, True, True, True, True, False], "<0.1 GFLOP Attempt 2 (Deep stack)")
    
    # 3. Stem at 16x16 and shrink channels locally.
    test_config([64, 128, 256, 256, 384, 384], [1, 2, 12, 15, 15, 11], [True, True, True, False, True, False], "<0.1 GFLOP Attempt 3 (Narrow Early Channels)")
