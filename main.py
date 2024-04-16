import torch
from torch import nn
import warnings

class TemporalLayer(nn.Module):
    @property
    def out_channels(self) -> int:
        return sum(self.kernels.values())
    
    def __init__(self, in_channels: int, kernels: dict) -> None:
        """Simple 1D convolutional network for temporal data.

        Args:
            in_channels (int): How many input channels the time series has.
            kernels (dict): A dict with as keys the size/length of each kernel and as value the number of kernels of that length.
        """
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
        """Simple 1D convolutional neural network for temporal data that only operates on the spatial dimension. Learns the interaction between each input channel at each, individual, time step.
        Useless on time series data with only one input channel.

        Args:
            in_channels (int): How many input channels the time series has.
            n_kernels (int): How many kernels to use.
        """
        super().__init__()
        self.output_size = n_kernels
        if in_channels < 1:
            warnings.warn(
                "Input of spatial layer has only 1 channel. \
                    Consider not using this layer."
            )
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
    def __init__(
        self,
        in_channels: int,
        temporal_kernels: dict,
        n_resolutions_learned: int,
        spatial_kernels: int = 0,
        dense: bool | str = False
    ) -> None:
        """A convolutional neural network for spatio-temporal data that intelligently learns what kernel size or resolution to focus on at each time step. This is done by following a temporal layer with a spatial layer whose sole job is to balance the outputs of the previous temporal layer. This temporal layer should use multiple kernel lengths, i.e. multiple resolutions.
        Optionally the first, temporal, layer can be preceded by a spatial layer that learns the interactions between each input channel.

        Args:
            in_channels (int): See `TemporalLayer`.
            temporal_kernels (dict): See `TemporalLayer`.
            n_resolutions_learned (int): How many kernels to use in the spatial/resolution layer. See `SpatialLayer`.
            spatial_kernels (int, optional): How many kernels to use in the optional, initial, spatial step. Defaults to 0, in which case no initial spatial step is used.
            dense (bool | str, optional) Can be one for True/False, 'append' or 'spatial'. When set to 'append' the original input is appended to the final output. When set to 'spatial' the original input is appended to after the optional first spatial layer. In this case spatial_kernels should be greater than 0. Defaults to False.
        """
        super().__init__()
        if spatial_kernels < 1 and dense == 'spatial':
            raise ValueError("Can't use dense='spatial' when spatial_kernels \
                is not set to an int greater than 0.")
        
        # init
        self.dense = dense
        self.convs = nn.Sequential()
        n_intermediary_channels = 0
        
        # optionally, prepend spatial layer
        if spatial_kernels > 0:
            self.convs.append(
                SpatialLayer(
                    in_channels=in_channels,
                    n_kernels=spatial_kernels,
                )
            )
            n_intermediary_channels = spatial_kernels
        if ( dense is True ) or ( dense == 'spatial' ):
            n_intermediary_channels += in_channels
        
        # learn over time
        temporal_layer = TemporalLayer(
            in_channels=n_intermediary_channels,
            kernels=temporal_kernels,
        )
        if dense is True:
            n_intermediary_channels += temporal_layer.out_channels
        else:
            n_intermediary_channels = temporal_layer.out_channels
        
        # learn to relate output channels of temporal layer (previous layer)
        resolution_learner = SpatialLayer(
            in_channels=n_intermediary_channels,
            n_kernels=n_resolutions_learned,
        )
        if dense is True:
            n_intermediary_channels += resolution_learner.output_size
        else:
            n_intermediary_channels = resolution_learner.output_size
        
        # store layers
        self.convs.extend(
            [
                temporal_layer,
                resolution_learner,
            ]
        )
        self.out_channels = n_resolutions_learned if dense is False \
            else n_intermediary_channels
        
        # choose a different forward function depending on dense
        self.forward = self._forward
        if self.dense == 'spatial':
            self.forward = self._forward_spatially_dense
        elif self.dense == 'append':
            self.forward = self._forward_append
        elif self.dense is True:
            self.forward = self._forward_dense
    
    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        y_ = self.convs(X)
        return y_
    
    def _forward_spatially_dense(self, X: torch.Tensor) -> torch.Tensor:
        """Append input X to output of first convolution."""
        y_ = self.convs[0](X)
        X_ = torch.concat([X, y_], dim=-2)
        y__ = self.convs[1:](X_)
        return y__
    
    def _forward_append(self, X: torch.Tensor) -> torch.Tensor:
        """Append input X to ouput."""
        y_ = self._forward(X)
        y_ = torch.concat([X, y_], dim=-2)
        return y_
    
    def _forward_dense(self, X: torch.Tensor) -> torch.Tensor:
        """Append each intermediate output to the output of the next layer and use this intermediate X_ as the input of each successive layer."""
        intermediates = [X]
        X_ = X
        for conv in self.convs:
            y_ = conv(X_)
            intermediates.append(y_)
            X_ = torch.concat(intermediates, dim=-2)
        return X_


if __name__ == "__main__":
    input = torch.rand(16, 4, 120)
    model = VarNet(4, {3: 2, 9: 2}, 3, 2, dense='append')
    # model = VarNet(4, {3: 2, 9: 2}, 3)
    output = model(input)
    print(output.shape)
