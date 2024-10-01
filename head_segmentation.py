from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torchvision.models.resnet import BasicBlock
from PIL import Image

@dataclass
class HeadSegmentationModelOutput(BaseOutput):
    sample: torch.Tensor

class SimpleResNetEncoder(nn.Module):
    def __init__(self, out_channels, depth=5, in_channels=3):
        super().__init__()
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels

        # Define layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._in_channels = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self._in_channels, out_channels, stride, downsample)]
        self._in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    def get_stages(self):
        return [
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = [x]
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)

        return features

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    

class HeadSegmentationModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        encoder_depth: int = 5,
        input_resolution: int = 512,
    ):
        super().__init__()

        block_out_channels = (3, 64, 64, 128, 256, 512)

        self.encoder = SimpleResNetEncoder(
            out_channels=block_out_channels,
            depth=encoder_depth,
            in_channels=in_channels
        )

        decoder_channels = [input_resolution // (2 ** i) for i in range(encoder_depth)]

        self.decoder = UnetDecoder(
            encoder_channels=block_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Sequentially pass `x` through model's encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return HeadSegmentationModelOutput(sample=masks)

class HeadSegmentationPipeline:
    def __init__(self, model_name_or_path: str, device: torch.device = torch.device('cpu')):
        self.device = device
        self.model = HeadSegmentationModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = image.resize((self.model.config.input_resolution, self.model.config.input_resolution))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)

    def postprocess(self, output: torch.Tensor, original_image: Image.Image) -> Image.Image:
        output = output.squeeze().argmax(dim=0).cpu().numpy().astype(np.uint8) * 255
        output = Image.fromarray(output)
        return output.resize(original_image.size, Image.Resampling.NEAREST)

    def __call__(self, image: Image.Image) -> Image.Image:
        preprocessed_image = self.preprocess(image)
        print(preprocessed_image.shape, preprocessed_image.dtype)
        with torch.no_grad():
            model_output = self.model(preprocessed_image)
        return self.postprocess(model_output.sample, original_image=image)
    
import argparse

def main(image_path: str, mask_path: str):
    pipeline = HeadSegmentationPipeline("okaris/head-segmentation")

    image = Image.open(image_path)
    mask = pipeline(image)
    mask.save(mask_path)
    print(f"Mask saved to {mask_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run head segmentation on the provided image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output mask")
    args = parser.parse_args()

    main(args.image_path, args.output_path)