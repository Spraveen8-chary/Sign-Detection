import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.boxes import rescale_bboxes, stacker
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
from utils.setup import get_classes, get_dataset_root, get_split_seed, get_train_split


class DETRData(Dataset):
    def __init__(self, path, train=True, split_ratio=0.85, split_seed=42):
        super().__init__()
        self.path = Path(path)
        self.train = train
        self.split_ratio = split_ratio
        self.split_seed = split_seed

        # Initialize logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()

        self.images_path, self.labels_path, split_source = self._resolve_paths()
        self.split_source = split_source
        self.labels = self._select_labels()

        # Log dataset initialization
        dataset_info = {
            "Dataset Path": str(self.path),
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": str(self.images_path),
            "Labels Path": str(self.labels_path),
            "Split Source": split_source,
            "Split Ratio": self.split_ratio,
            "Split Seed": self.split_seed,
        }
        self.data_handler.log_dataset_stats(dataset_info)

        # Log transforms information
        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor"
        ]
        self.data_handler.log_transform_info(transform_list)

    def _resolve_paths(self):
        train_images = self.path / "train" / "images"
        train_labels = self.path / "train" / "labels"
        test_images = self.path / "test" / "images"
        test_labels = self.path / "test" / "labels"

        # Unified mode: data_root/images + data_root/labels
        images = self.path / "images"
        labels = self.path / "labels"
        if images.exists() and labels.exists() and any(labels.glob("*.txt")):
            return images, labels, "in_memory_split"

        # Backward-compatible split-directory mode
        if train_images.exists() and train_labels.exists() and test_images.exists() and test_labels.exists():
            if self.train:
                return train_images, train_labels, "folder_split(train)"
            return test_images, test_labels, "folder_split(test)"

        # If unified folders exist but are still empty, keep unified mode.
        if images.exists() and labels.exists():
            return images, labels, "in_memory_split(empty)"

        raise FileNotFoundError(
            f"Dataset not found under {self.path}. Expected either "
            f"train/test split folders or unified images/labels."
        )

    def _select_labels(self):
        label_files = sorted([f for f in os.listdir(self.labels_path) if f.endswith(".txt")])
        if len(label_files) == 0:
            return []

        # If we are in explicit train/test directories, keep all labels in that folder
        if self.split_source.startswith("folder_split"):
            return label_files

        # Unified mode: deterministic in-memory split
        if len(label_files) == 1:
            return label_files if self.train else []

        indices = np.arange(len(label_files))
        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(indices)

        split_idx = int(len(label_files) * self.split_ratio)
        split_idx = max(1, min(len(label_files) - 1, split_idx))
        chosen = indices[:split_idx] if self.train else indices[split_idx:]
        selected = [label_files[i] for i in chosen]
        selected.sort()
        return selected

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        self.transform = A.Compose(
            [   
                A.Resize(500,500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []), # Example random crop
                A.Resize(224,224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *([A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)] if self.train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

        # Fallback path that always returns a tensor image.
        self.fallback_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        
        for attempt in range(max_attempts):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                # Check if we still have bboxes after transformation
                if len(transformed['bboxes']) > 0:
                    return transformed
            except:
                continue
        
        fallback = self.fallback_transform(image=image)
        return {"image": fallback["image"], "bboxes": bboxes, "class_labels": labels}

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        self.label_path = os.path.join(self.labels_path, self.labels[idx])
        self.image_name = self.labels[idx].split('.')[0]
        self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg')

        img = Image.open(self.image_path)
        with open(self.label_path, 'r') as f:
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations:
            parts = annotation.strip().split(" ")
            if len(parts) != 5:
                continue
            class_labels.append(parts[0])
            bounding_boxes.append(parts[1:])

        if len(class_labels) == 0:
            raise ValueError(f"No valid annotations found in {self.label_path}")

        class_labels = np.array(class_labels).astype(int) 
        bounding_boxes = np.array(bounding_boxes).astype(float) 

        augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
        augmented_img_tensor = augmented['image']
        augmented_bounding_boxes = np.array(augmented['bboxes'])
        augmented_classes = augmented['class_labels']

        if isinstance(augmented_img_tensor, np.ndarray):
            # Defensive conversion in case any transform path returns ndarray.
            if augmented_img_tensor.ndim == 3:
                augmented_img_tensor = torch.from_numpy(augmented_img_tensor).permute(2, 0, 1).float()
            else:
                augmented_img_tensor = torch.from_numpy(augmented_img_tensor).float()

        labels = torch.tensor(augmented_classes, dtype=torch.long)  
        boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
        return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

if __name__ == '__main__':
    dataset = DETRData(get_dataset_root(), train=True, split_ratio=get_train_split(), split_seed=get_split_seed())
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET) 
    CLASSES = get_classes() 
    fig, ax = plt.subplots(2,2) 
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
        ax.imshow(img.permute(1,2,0))
        box_classes = annotations['labels'] 
        boxes = rescale_bboxes(annotations['boxes'], (224,224))
        for box_class, bbox in zip(box_classes, boxes):
            xmin, ymin, xmax, ymax = bbox.detach().numpy()
            print(xmin, ymin, xmax, ymax) 
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
            text = f'{CLASSES[box_class]}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    fig.tight_layout() 
    plt.show()     
