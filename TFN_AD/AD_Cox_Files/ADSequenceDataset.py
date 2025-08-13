from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

class ADSequenceDataset(Dataset):
    """
    Dataset that takes the Step 5 output (a list of PatientSeq) and returns
    per-patient sequences.

    __getitem__(i) -> dict:
      - x_struct: FloatTensor [T, F]
      - x_time:   FloatTensor [T]
      - x_img:    FloatTensor [T, 3, 224, 224]  (when images are loaded) / None (when images are disabled)
      - event:    FloatTensor []  (scalar)
      - pid:      str
    """
    def __init__(self, seqs: List["PatientSeq"], img_transform=None, load_images: bool = True):
        """
        Args:
            seqs: List of PatientSeq processed up through Step 6
            img_transform: e.g., torchvision.transforms.Compose (to convert images to 224x224 tensors)
            load_images: True to load images; False to return x_img=None
        """
        self.seqs = seqs
        self.img_transform = img_transform
        self.load_images = load_images

        # Ensure all samples consistently have images or not
        path_lens = [len(s.img_paths) for s in seqs]
        has_any = any(l > 0 for l in path_lens)
        has_none = any(l == 0 for l in path_lens)
        if load_images and has_any and has_none:
            raise ValueError(
                "Some sequences have images while others do not. "
                "Rebuild with require_images=True in Step 5 or set load_images=False."
            )
        self.has_images = load_images and has_any

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.seqs[i]

        # Structured features / time / label
        x_struct = torch.from_numpy(s.struct).float()    # [T,F]
        x_time   = torch.from_numpy(s.times).float()     # [T]
        event    = torch.tensor(float(s.event), dtype=torch.float32)  # []

        # Images (optional)
        x_img: Optional[torch.Tensor] = None
        if self.has_images:
            if self.img_transform is None:
                raise ValueError("img_transform must be provided when load_images=True.")
            imgs = []
            for p in s.img_paths:
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        imgs.append(self.img_transform(im))
                except Exception as e:
                    # If image loading fails, replace with a zero-tensor (size 3x224x224)
                    print(f"[WARN] bad image: {p} ({e})")
                    imgs.append(torch.zeros(3, 224, 224, dtype=torch.float32))
            x_img = torch.stack(imgs, dim=0)             # [T,3,224,224]

        return {
            "x_struct": x_struct,
            "x_time": x_time,
            "x_img": x_img,
            "event": event,
            "t_event": torch.tensor(float(s.t_event), dtype=torch.float32),  # ★ added
            "pid": s.pid,
        }
