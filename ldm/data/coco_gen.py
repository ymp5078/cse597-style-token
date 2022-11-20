import os
import numpy as np
import PIL
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms


class COCOGenBase(Dataset):
    def __init__(self,
                 json_path,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 **kwargs
                 ):
        self.json_path = json_path
        self.data_root = data_root
        with open(self.json_path, "r") as f:
            cg_index = json.load(f)
        self.image_captions = [(os.path.join(self.data_root,str(index['id']),img),index['caption']) for index in cg_index for img in os.listdir(os.path.join(self.data_root,str(index['id']))) ]
        self._length = len(self.image_captions)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path, cap = self.image_captions[i]
        # example = dict((k, self.labels[k][i]) for k in self.labels)
        example = {}
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example['caption'] = cap
        return example


class COCOGenTrain(COCOGenBase):
    def __init__(self, **kwargs):
        super().__init__(json_path="/home/ymp5078/projects/datasets/coco/coco_gen.json", data_root="/home/ymp5078/projects/video_gen/stable-diffusion/outputs/coco_gen", **kwargs)


class COCOGenValidation(COCOGenBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_path="/home/ymp5078/projects/datasets/coco/coco_gen.json", data_root="/home/ymp5078/projects/video_gen/stable-diffusion/outputs/coco_gen",
                         flip_p=flip_p, **kwargs)


