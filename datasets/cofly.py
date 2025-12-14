import io
import os.path

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from patchify import patchify
from skimage import io as skio
from PIL import Image
from sklearn.model_selection import train_test_split

from datasets.augmentations_geometry import *
from datasets.augmentations_normalize import *
from datasets.augmentations_color import *


def get_labeled_image_paths(img_path: str, anno_path: str, exts=('.jpg', '.png', '.jpeg')):
    """
    Return two list, image and labels.
    """
    image_paths = []
    label_paths = []
    anno_files = sorted(os.listdir(anno_path))

    for anno_file in anno_files:
        image_paths.append(os.path.join(img_path, anno_file))
        label_paths.append(os.path.join(anno_path, anno_file))

    return image_paths, label_paths


class Patches:
    def __init__(self, im_list, msk_list, patch_size: int = 256, threshold=0.03):
        self.im_list = im_list
        self.msk_list = msk_list
        self.patch_size = patch_size
        self.threshold = threshold

    def image_to_patches(self, image, b_msk=False):
        slc_size = self.patch_size
        x = int(math.ceil(int(image.shape[0]) / (slc_size * 1.0)))  # /256 ->
        y = int(math.ceil(int(image.shape[1]) / (slc_size * 1.0)))
        padded_shape = (x * slc_size, y * slc_size)
        if not b_msk:
            padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1], 3), dtype=np.uint8)
            padded_rgb_image[:image.shape[0], :image.shape[1]] = image
            patches = patchify(padded_rgb_image, (slc_size, slc_size, 3), step=slc_size)
        elif b_msk:
            padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1]), dtype=np.uint8) * 255
            padded_rgb_image[:image.shape[0], :image.shape[1]] = image
            patches = patchify(padded_rgb_image, (slc_size, slc_size), step=slc_size)

        return patches, slc_size

    def load_image(self, path):
        """
        loads an image based on the path
        """
        rgb_image = skio.imread(path)

        return rgb_image

    def patchify_image_mask(self):
        imgs = []
        anns = []
        names = []  # new!
        AREA = self.patch_size * self.patch_size
        f_AREA = int(self.threshold * AREA)
        print(f'Threshold: {self.threshold} * {AREA} = {f_AREA}')
        print(f"Patchyfying images and mask...")
        for im_path, msk_path in zip(self.im_list, self.msk_list):
            patches, _ = self.image_to_patches(self.load_image(im_path))  # 这里加载图像使用skimage返回numpy.ndarray类型，便于后处理
            masks, _ = self.image_to_patches(self.load_image(msk_path), b_msk=True)
            base_name = os.path.splitext(os.path.basename(im_path))[0]
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, :, :, :]
                    mask = masks[i, j, ::]
                    if mask.reshape(-1).sum() > f_AREA:
                        imgs.append(patch)
                        anns.append(mask)
                        names.append(f'{base_name}_{i}_{j}.png')  # new!
        return np.array(imgs), np.array(anns), names


class CoFLY(Dataset):
    """ Represents the CoFly dataset.

    The directory structure is as following:
    ├── images    (# 366)
    ├── labels_1d (# 201)
    """

    def __init__(self,
                 path_to_dataset: str,
                 mode: str,
                 img_normalizer: ImageNormalizer,
                 augmentations_geometric: List[GeometricDataAugmentation],
                 augmentations_color: List[Callable],
                 patch_size: int = 256):
        assert os.path.exists(path_to_dataset), f'The path to dataset does not exist: {path_to_dataset}.'
        super(CoFLY, self).__init__()

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.img_normalizer = img_normalizer
        self.augmentations_geometric = augmentations_geometric
        self.augmentations_color = augmentations_color
        self.patch_size = patch_size

        # collect paired images with labels
        self.path_to_images = os.path.join(path_to_dataset, 'images')
        self.path_to_annos = os.path.join(path_to_dataset, 'labels_1d')

        self.image_paths, self.anno_paths = get_labeled_image_paths(self.path_to_images, self.path_to_annos)

        # patchifying  image_shape[787, 1 256, 256, 3] anno_shape[787, 256, 256]
        self.image_data, self.anno_data, self.image_filenames = Patches(self.image_paths,
                                                                        self.anno_paths).patchify_image_mask()  # dtype uint8

        print(np.unique(self.anno_data))
        # Reshape the anno_shape have a channel-dimension, and strip the '1' dimension in the image_data arrays.
        self.image_data = np.squeeze(self.image_data)
        self.anno_data = np.expand_dims(self.anno_data, axis=-1)

        print(f'Image_shape = {self.image_data.shape}\nanno_shape = {self.anno_data.shape}')
        print(f'n_classes: {len(np.unique(self.anno_data))}')

        split_idx = int(0.8 * len(self.image_data))
        self.train_images = self.image_data[:split_idx]
        self.train_annos = self.anno_data[:split_idx]
        self.filenames_train = self.image_filenames[:split_idx]
        self.val_images = self.image_data[split_idx:]
        self.val_annos = self.anno_data[split_idx:]
        self.filenames_val = self.image_filenames[split_idx:]

        # split_idx = int(0.2 * len(self.image_data))
        # self.val_images = self.image_data[:split_idx]
        # self.val_annos = self.anno_data[:split_idx]
        # self.filenames_val = self.image_filenames[:split_idx]
        # self.train_images = self.image_data[split_idx:]
        # self.train_annos = self.anno_data[split_idx:]
        # self.filenames_train = self.image_filenames[split_idx:]


        print(f'Train_images = {len(self.train_images)}\nVal_images = {len(self.val_images)}')
        # ----------------------------------------Prepare Testing ------------------------------------------------------
        self.path_to_test_images = os.path.join(path_to_dataset, 'test', 'images')
        self.path_to_test_annos = os.path.join(path_to_dataset, 'test', 'semantics')
        self.filenames_test = get_img_fnames_in_dir(self.path_to_test_images)

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()

    def get_train_item(self, idx: int) -> Dict:
        img_skio = self.train_images[idx]
        img = self.img_to_tensor(img_skio)

        if random.random() > 0.25:
            for augmentations_color_fn in self.augmentations_color:
                img = augmentations_color_fn(img)

        anno = self.train_annos[idx]
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = anno.astype(np.int64)  # [H x W]
        anno = torch.Tensor(anno).type(torch.int64)
        anno = anno.unsqueeze(0)

        for augmentor_geometric_fn in self.augmentations_geometric:
            img, anno = augmentor_geometric_fn(img, anno)
        anno = anno.squeeze(0)  # [H x W]

        mask_2 = anno == 2
        anno[mask_2] = 1

        mask_3 = anno == 3
        anno[mask_3] = 1

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        return {'input_image_before_norm': img_before_norm,
                'input_image': img,
                'anno': anno,
                'fname': self.filenames_train[idx]}

    def get_val_item(self, idx: int) -> Dict:
        img_skio = self.val_images[idx]
        img = self.img_to_tensor(img_skio)

        anno = self.val_annos[idx]
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = anno.astype(np.int64)  # [H x W]
        anno = torch.Tensor(anno).type(torch.int64)

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        mask_2 = anno == 2
        anno[mask_2] = 1

        mask_3 = anno == 3
        anno[mask_3] = 1

        return {'input_image_before_norm': img_before_norm,
                'input_image': img,
                'anno': anno,
                'fname': self.filenames_val[idx]}

    def get_test_item(self, idx: int) -> Dict:
        path_to_current_img = os.path.join(self.path_to_test_images, self.filenames_test[idx])
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)

        path_to_current_anno = os.path.join(self.path_to_test_annos, self.filenames_test[idx])
        anno = np.array(Image.open(path_to_current_anno))
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = anno.astype(np.int64)
        anno = torch.Tensor(anno).type(torch.int64)

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        mask_3 = anno == 3
        anno[mask_3] = 1

        mask_4 = anno == 4
        anno[mask_4] = 2

        return {'input_image_before_norm': img_before_norm,
                'input_image': img,
                'anno': anno}

    def __getitem__(self, idx: int):
        if self.mode == 'train':
            items = self.get_train_item(idx)
            return items

        if self.mode == 'val':
            items = self.get_val_item(idx)
            return items

        if self.mode == 'test':
            items = self.get_test_item(idx)
            return items

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images)

        if self.mode == 'val':
            return len(self.val_images)

        if self.mode == 'test':
            return len(self.filenames_test)


class CoFlyModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict):
        super(CoFlyModule, self).__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        path_to_dataset = self.cfg['data']['path_to_dataset']
        image_normalizer = get_image_normalizer(self.cfg)

        if stage == 'fit' or stage == 'validate' or stage is None:
            # -----------------------------------------TRAIN-----------------------------------------------------------
            train_augmentations_geometric = get_geometric_augmentations(self.cfg, 'train')
            train_augmentations_color = get_color_augmentations(self.cfg, 'train')
            self.train_ds = CoFLY(
                path_to_dataset,
                mode='train',
                img_normalizer=image_normalizer,
                augmentations_geometric=train_augmentations_geometric,
                augmentations_color=train_augmentations_color)

            # -----------------------------------------VAL--------------------------------------------------------------
            val_augmentations_geometric = get_geometric_augmentations(self.cfg, 'val')
            val_augmentations_color = get_color_augmentations(self.cfg, 'val')
            self.val_ds = CoFLY(
                path_to_dataset,
                mode='val',
                img_normalizer=image_normalizer,
                augmentations_geometric=val_augmentations_geometric,
                augmentations_color=[])

        # ------------------------------------------TEST-------------------------------------------------------------
        if stage == 'test' or stage is None:
            test_augmentations_geometric = get_geometric_augmentations(self.cfg, 'test')
            self.test_ds = CoFLY(
                path_to_dataset,
                mode='test',
                img_normalizer=image_normalizer,
                augmentations_geometric=test_augmentations_geometric,
                augmentations_color=[])

    def train_dataloader(self) -> DataLoader:
        # Return Dataloader for Training Data here
        shuffle: bool = self.cfg['train']['shuffle']
        batch_size: int = self.cfg['train']['batch_size']
        n_workers: int = self.cfg['data']['num_workers']

        loader = DataLoader(self.train_ds,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=n_workers,
                            drop_last=True,
                            pin_memory=True)
        return loader

    def val_dataloader(self) -> DataLoader:
        batch_size: int = self.cfg['val']['batch_size']
        n_workers: int = self.cfg['data']['num_workers']

        loader = DataLoader(self.val_ds,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
        return loader

    def test_dataloader(self) -> DataLoader:
        batch_size: int = self.cfg['test']['batch_size']
        n_workers: int = self.cfg['data']['num_workers']

        loader = DataLoader(self.test_ds,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            shuffle=False,
                            pin_memory=True)
        return loader


# if __name__ == '__main__':
#     cofly = CoFLY(path_to_dataset='/home/cfh/DataSet/CoFly-WeedDB',
#                   mode='val',
#                   img_normalizer=SingleImageNormalizer(),
#                   augmentations_geometric=[],
#                   augmentations_color=[]).get_train_item
#     print(cofly)
