import torch
from torch import nn


class TemporalLayer(nn.Module):
    @property
    def out_channels(self) -> int:
        return sum(self.kernels.values())
    
    def __init__(self, in_channels: int, kernels: dict) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernels = kernels
        
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


class SpatialLayer(nn.Module):
    def __init__(self, in_channels: int, n_kernels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_kernels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
    
    def  forward(self, X: torch.Tensor) -> torch.Tensor:
        y_ = self.conv(X)
        return y_


class VarNet(nn.Module):
    @property
    def out_channels(self) -> int:
        return self.spatial_kernels
    
    def __init__(
        self,
        in_channels: int,
        temporal_kernels: dict,
        spatial_kernels: int,
    ) -> None:
        super().__init__()
        self.spatial_kernels = spatial_kernels
        
        temporal_layer = TemporalLayer(
            in_channels=in_channels,
            kernels=temporal_kernels,
        )
        spatial_layer = SpatialLayer(
            in_channels=temporal_layer.out_channels,
            n_kernels=spatial_kernels,
        )
        self.conv = nn.Sequential(
            temporal_layer,
            spatial_layer
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y_ = self.conv(X)
        return y_


if __name__ == "__main__":
    input = torch.rand(16, 4, 120)
    model = VarNet(4, {3: 2, 9: 2}, 3)
    output = model(input)
    print(output.shape)
