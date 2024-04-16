import torch
from torch import nn


class VarNet(nn.Module):
    def __init__(self, in_channels: int, kernels: dict) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        
        for kernel_length, kernel_count in kernels.items():
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=kernel_count,
                    kernel_size=kernel_length,
                    stride=1,
                    padding='same',
                    dilation=1,
                    bias=False,
                )
            )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y_ = [conv(X) for conv in self.convs]
        y_ = torch.concat(y_, dim=-2)
        return y_


if __name__ == "__main__":
    input = torch.rand(16, 4, 120)
    model = VarNet(4, {3: 2, 9: 2})
    output = model(input)
    print(output.shape)