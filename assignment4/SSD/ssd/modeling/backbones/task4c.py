import torch
from typing import Tuple, List


class ImprovedModel(torch.nn.Module):
    """
    This is an experimental improvement model from the basic SSD model.
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        num_filters = 64
        self.feature_map_zero = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.BatchNorm2d(num_features=num_filters),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(num_filters, num_filters * 2, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=num_filters * 2),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(num_filters * 2, num_filters * 2, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=num_filters * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_filters * 2, self.out_channels[0], 3, 2, 1)
        )

        self.feature_map_one = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channels[0], 128, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.out_channels[1], 3, 2, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[1]),
        )

        self.feature_map_two = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channels[1], 256, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, self.out_channels[2], 3, 2, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[2]),
        )

        self.feature_map_three = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channels[2], 128, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[2]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.out_channels[3], 3, 2, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[3]),
        )

        self.feature_map_four = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channels[3], 128, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[3]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.out_channels[4], 3, 2, 1),
            torch.nn.BatchNorm2d(num_features=self.out_channels[4]),
        )

        self.feature_map_five = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channels[4], 128, 3, 1, 1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.out_channels[5], 3, 1, 0),
        )
        
        self.feature_maps = [
            self.feature_map_zero,
            self.feature_map_one,
            self.feature_map_two,
            self.feature_map_three,
            self.feature_map_four,
            self.feature_map_five,
        ]

    def forward(self, x):
        """
        The forward function
        """
        out_feats = []

        out_feats.append(self.feature_map_zero(x))
        for i in range(1, len(self.feature_maps)):
            out_feats.append(self.feature_maps[i](out_feats[i - 1]))

        for idx, feature in enumerate(out_feats):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_feats) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_feats)}"
        return tuple(out_feats)

