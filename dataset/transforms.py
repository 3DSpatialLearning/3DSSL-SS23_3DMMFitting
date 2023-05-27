from dataclasses import dataclass, field
import torch
from torchvision import transforms


@dataclass
class ToTensor:
    to_tensor: transforms.ToTensor = field(default_factory=transforms.ToTensor)
    normalize: bool = True

    def __call__(self, features: dict) -> dict:
        features["image"] = self.to_tensor(features["image"])  # H x W x C -> C x H x W
        if not self.normalize:
            features["image"] *= 255.
        features["consistency"] = torch.from_numpy(features["consistency"])
        features["depth"] = torch.from_numpy(features["depth"])
        features["normal"] = torch.from_numpy(features["normal"])
        features["intrinsics"] = torch.from_numpy(features["intrinsics"])
        features["extrinsics"] = torch.from_numpy(features["extrinsics"])
        if "gt_landmark" in features:
            features["gt_landmark"] = torch.from_numpy(features["gt_landmark"])
        if "predicted_landmark" in features:
            features["predicted_landmark"] = torch.from_numpy(features["predicted_landmark"])
        return features
