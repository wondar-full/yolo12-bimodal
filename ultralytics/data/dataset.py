# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return dictionary of labels for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Set mosaic, copy_paste and mixup options to 0.0 and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # 🆕 计算目标面积 (for VisDrone size-wise metrics)
        if len(bboxes) > 0:
            if bbox_format == "xyxy":
                # bboxes格式: [x1, y1, x2, y2, ...]
                widths = bboxes[:, 2] - bboxes[:, 0]
                heights = bboxes[:, 3] - bboxes[:, 1]
            elif bbox_format == "xywh":
                # bboxes格式: [x_center, y_center, w, h, ...]
                widths = bboxes[:, 2]
                heights = bboxes[:, 3]
            else:
                # 其他格式暂不支持,设为0
                widths = np.zeros(len(bboxes))
                heights = np.zeros(len(bboxes))
            
            # 🔧 Bug Fix: 如果是归一化坐标,需要乘以图像尺寸才能得到像素面积
            if normalized:
                # 优先使用resized_shape (验证时bbox是相对于resize后的尺寸,而非原始尺寸)
                img_h, img_w = label.get("resized_shape", (640, 640))[:2]
                
                # 如果resized_shape不存在,尝试从img获取实际尺寸
                if "img" in label and label["img"] is not None:
                    img_h, img_w = label["img"].shape[:2]
                
                widths = widths * img_w
                heights = heights * img_h
            
            target_areas = (widths * heights).astype(np.float32)
        else:
            target_areas = np.array([], dtype=np.float32)
        
        label["target_areas"] = target_areas  # 🆕 添加到label字典

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            # 🆕 target_areas 需要concat (与bboxes/cls一样)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "target_areas"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLORGBDDataset(YOLODataset):
    """
    Dataset class for loading RGB-D (RGB + Depth) object detection labels in YOLO format.
    
    This class extends YOLODataset to support dual-modal input by automatically loading
    paired depth images alongside RGB images. Depth images are preprocessed, normalized,
    and concatenated with RGB channels to form 4-channel tensors [R, G, B, D].
    
    Key Features:
        - Automatic RGB-Depth pairing based on data.yaml configuration
        - Robust depth preprocessing (median filter → gaussian smooth → confidence weighting)
        - Graceful fallback to RGB-only mode if depth missing
        - Support for different depth formats (16-bit PNG, 8-bit JPG, TIFF)
        
    Data.yaml Format:
        ```yaml
        path: /path/to/dataset
        train: images/train
        val: images/val
        train_depth: depths/train  # Add this for depth support
        val_depth: depths/val      # Add this for depth support
        ```
    
    Attributes:
        depth_files (list[str] | None): List of depth image paths paired with im_files.
        _depth_enabled (bool): Whether depth modality is active.
        _depth_split (str | None): Current data split ('train'/'val'/'test').
        _depth_pairs (list[tuple[Path, Path]]): RGB root to Depth root mappings.
        
    Methods:
        load_image: Override to load and fuse RGB + Depth channels.
        _initialize_depth_paths: Build RGB-to-Depth path mapping.
        _process_depth_channel: Preprocess depth map (denoise, normalize, weight).
        
    Examples:
        >>> # Basic usage
        >>> data_yaml = {"train": "images/train", "train_depth": "depths/train"}
        >>> dataset = YOLORGBDDataset(img_path="train", data=data_yaml)
        >>> img, _, _ = dataset.load_image(0)
        >>> print(img.shape)  # (H, W, 4) - RGB+D
        
        >>> # Batch loading
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=16, collate_fn=YOLODataset.collate_fn)
        >>> batch = next(iter(loader))
        >>> print(batch['img'].shape)  # [B, 4, H, W]
        
    📚 八股知识点: RGB-D数据加载
    Q: 为什么要单独加载深度图而不是预先拼接？
    A: (1) 灵活性: 可以动态调整深度预处理策略
       (2) 存储效率: 深度通常是16-bit，预拼接会浪费空间
       (3) 数据增强: RGB和Depth需要同步增强（旋转、裁剪）
       (4) 调试方便: 可以单独可视化RGB和Depth
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Initialize RGB-D dataset with automatic depth pairing.
        
        Args:
            data (dict, optional): Dataset configuration with optional depth paths.
            task (str): Task type ('detect', 'segment', 'pose', 'obb').
            *args: Positional arguments for parent YOLODataset.
            **kwargs: Keyword arguments for parent YOLODataset.
            
        Note:
            If data.yaml contains 'train_depth' or 'val_depth', depth modality
            will be enabled automatically. Otherwise, operates as RGB-only.
        """
        # RGB-D specific attributes
        self.depth_files: list[str] | None = None
        self._depth_enabled: bool = False
        self._depth_split: str | None = None
        self._depth_pairs: list[tuple[Path, Path]] = []
        self._input_img_paths: list[str | Path] = []
        
        # Store original img_path before super().__init__
        if args:
            self._input_img_paths = self._ensure_list(args[0])
        elif 'img_path' in kwargs:
            self._input_img_paths = self._ensure_list(kwargs['img_path'])
        
        # Initialize parent YOLODataset (loads RGB images)
        super().__init__(*args, data=data, task=task, **kwargs)
        
        # Build depth pairing after RGB images are loaded
        self._initialize_depth_paths()

    @staticmethod
    def _ensure_list(x: Any) -> list:
        """Convert input to list if not already."""
        if x is None:
            return []
        return x if isinstance(x, (list, tuple)) else [x]

    def _has_depth_config(self) -> bool:
        """Check if data.yaml specifies depth paths."""
        if not self.data:
            return False
        return any(self.data.get(f"{split}_depth") for split in ("train", "val", "test"))

    def _initialize_depth_paths(self) -> None:
        """
        Build lookup table mapping RGB image paths to paired depth images.
        
        This method:
        1. Infers current data split (train/val/test)
        2. Resolves depth root directories from data.yaml
        3. Matches each RGB image to its corresponding depth image
        4. Validates all pairs exist (raises error if missing)
        
        Raises:
            FileNotFoundError: If depth images are missing for some RGB samples.
            
        📚 八股知识点: 文件路径匹配
        Q: 为什么需要directory pair映射？
        A: (1) 支持不同的目录结构(depths/和images/可能不在同一父目录)
           (2) 支持多个数据源拼接
           (3) 相对路径转绝对路径的鲁棒处理
           (4) 自动扩展名匹配(.png, .jpg, .tiff等)
        """
        if not self._has_depth_config() or not self.im_files:
            return

        # Infer which split this dataset represents
        split = self._infer_depth_split()
        if split is None:
            return

        # Get depth directory specification
        depth_entry = self.data.get(f"{split}_depth")
        if not depth_entry:
            return

        # Collect RGB and Depth root directories
        rgb_roots = self._collect_existing_dirs(self._input_img_paths)
        depth_roots = self._resolve_modal_roots(depth_entry)
        
        if not rgb_roots:
            LOGGER.warning(f"{self.prefix}Unable to locate RGB root directories for depth alignment.")
            return
        if not depth_roots:
            LOGGER.warning(f"{self.prefix}Unable to locate depth root directories defined in data.yaml.")
            return
        
        # Broadcast single depth root to multiple RGB roots if needed
        if len(depth_roots) == 1 and len(rgb_roots) > 1:
            depth_roots = depth_roots * len(rgb_roots)
        
        if len(rgb_roots) != len(depth_roots):
            LOGGER.warning(
                f"{self.prefix}Mismatched RGB ({len(rgb_roots)}) and depth ({len(depth_roots)}) roots; depth disabled."
            )
            return

        # Build RGB→Depth directory pairs
        self._depth_pairs = list(zip(rgb_roots, depth_roots))
        
        # Match each RGB image to its depth image
        depth_files: list[str] = []
        missing: list[str] = []
        
        for im_path_str in self.im_files:
            depth_path = self._match_depth_path(Path(im_path_str))
            if depth_path is None or not depth_path.exists():
                missing.append(im_path_str)
            else:
                depth_files.append(str(depth_path))

        # Validate all pairs exist
        if missing:
            example = missing[0]
            raise FileNotFoundError(
                f"Depth image not found for {len(missing)} samples. Example missing pair: '{example}'.\n"
                "Ensure depth files mirror the RGB directory structure and filenames.\n"
                "Expected format: images/train/img001.jpg → depths/train/img001.png"
            )

        # Enable depth modality
        if depth_files:
            self.depth_files = depth_files
            self._depth_enabled = True
            self._depth_split = split
            LOGGER.info(
                f"{self.prefix}Depth modality enabled on split '{split}' with {len(self.depth_files)} paired samples."
            )

    def _infer_depth_split(self) -> str | None:
        """
        Infer which dataset split (train/val/test) this instance corresponds to.
        
        Returns:
            str | None: 'train', 'val', 'test', or None if cannot infer.
            
        Note:
            This compares the input img_path with data.yaml's train/val/test entries.
        """
        if not self.data:
            return None

        # Collect all resolved input paths
        resolved_inputs = set()
        for path in self._input_img_paths:
            if not path:
                continue
            try:
                resolved_inputs.add(Path(path).resolve())
            except Exception:
                continue

        # Match against data.yaml splits
        for split in ("train", "val", "test"):
            for candidate in self._ensure_list(self.data.get(split)):
                if not candidate:
                    continue
                try:
                    if Path(candidate).resolve() in resolved_inputs:
                        return split
                except Exception:
                    continue
        return None

    def _resolve_modal_roots(self, entries: Any) -> list[Path]:
        """
        Resolve modal-specific root paths relative to dataset root when necessary.
        
        Args:
            entries: Single path or list of paths (str/Path/None).
            
        Returns:
            list[Path]: List of resolved, existing directory paths.
            
        Example:
            data.yaml: path: /data/visdrone, train_depth: depths/train
            → Resolves to /data/visdrone/depths/train
        """
        roots: list[Path] = []
        base_path = None
        
        # Get dataset root path
        if self.data and self.data.get("path"):
            try:
                base_path = Path(self.data["path"]).resolve()
            except Exception:
                base_path = None

        # Resolve each entry
        for item in self._ensure_list(entries):
            candidate: Path | None = None
            try:
                path_obj = Path(item)
            except Exception:
                continue

            # Handle absolute vs relative paths
            if path_obj.is_absolute():
                candidate = path_obj
            elif base_path is not None:
                candidate = base_path / path_obj
            else:
                candidate = path_obj

            # Validate existence
            try:
                resolved = candidate.resolve()
            except Exception:
                continue

            if resolved.exists():
                roots.append(resolved)
                
        return roots

    @staticmethod
    def _collect_existing_dirs(paths: Any) -> list[Path]:
        """
        Return list of existing directory paths from iterable of path-like inputs.
        
        Args:
            paths: Iterable of path strings/objects or None values.
            
        Returns:
            list[Path]: List of existing, resolved directory paths.
        """
        collected: list[Path] = []
        for path in YOLORGBDDataset._ensure_list(paths):
            if not path:
                continue
            try:
                p = Path(path).resolve()
            except Exception:
                continue
            if p.exists():
                collected.append(p)
        return collected

    def _match_depth_path(self, image_path: Path) -> Path | None:
        """
        Locate corresponding depth image path given an RGB image path.
        
        Args:
            image_path: Path object of RGB image.
            
        Returns:
            Path | None: Matched depth image path, or None if not found.
            
        Matching Strategy:
        1. Compute relative path from RGB root
        2. Apply same relative path to Depth root
        3. Try exact match first
        4. Try glob match with different extensions (.png, .jpg, .tiff)
        
        Example:
            RGB: /data/images/train/folder/img001.jpg
            RGB root: /data/images/train
            Depth root: /data/depths/train
            → Try: /data/depths/train/folder/img001.png (exact)
            → Try: /data/depths/train/folder/img001.* (glob)
        """
        for rgb_root, depth_root in self._depth_pairs:
            try:
                # Get relative path from RGB root
                rel = image_path.relative_to(rgb_root)
            except ValueError:
                continue

            # Build candidate depth path
            candidate = depth_root / rel
            
            # Try exact match
            if candidate.exists():
                return candidate
            
            # Try glob match (different extensions)
            parent = candidate.parent
            if parent.exists():
                matches = list(parent.glob(candidate.stem + ".*"))
                for match in matches:
                    if match.exists():
                        return match
                        
        return None

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load RGB image and append paired depth channel when available.
        
        Args:
            i (int): Image index.
            rect_mode (bool): Whether to use rectangular inference mode.
            
        Returns:
            tuple: (image, original_hw, resized_hw)
                - image: [H, W, 4] numpy array (RGB+D) or [H, W, 3] (RGB-only)
                - original_hw: (height, width) before any transforms
                - resized_hw: (height, width) after letterbox/resize
                
        Processing Pipeline:
        1. Load RGB image via parent class (handles caching, letterbox)
        2. If depth enabled, load paired depth image
        3. Preprocess depth (denoise → normalize → weight by confidence)
        4. Resize depth to match RGB dimensions
        5. Concatenate [RGB, Depth] along channel axis
        6. Update cache if enabled
        
        📚 八股知识点: 深度图预处理
        Q: 为什么深度图需要median+gaussian双重滤波？
        A: (1) Median filter: 去除椒盐噪声(传感器噪点)
           (2) Gaussian filter: 平滑边缘，填补小范围缺失
           (3) 置信度加权: 抑制不可信区域(反光、透明物体)
           (4) 百分位拉伸: 自适应到不同场景的深度范围
        """
        # Load RGB image from parent class
        im, hw0, hw = super().load_image(i, rect_mode)
        
        # Return early if depth not enabled or already fused
        if not self._depth_enabled or (im.ndim == 3 and im.shape[2] == self.channels):
            return im, hw0, hw

        # Get depth file path
        depth_path = Path(self.depth_files[i]) if self.depth_files else None
        if depth_path is None:
            raise FileNotFoundError("Depth file list is empty; cannot fuse depth channel.")

        # Load depth image (supports 16-bit PNG, 8-bit JPG, TIFF)
        from ultralytics.data.loaders import imread
        depth = imread(str(depth_path), flags=cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        # Preprocess depth to match RGB dimensions
        depth = self._process_depth_channel(depth, im.shape[:2])

        # Concatenate RGB + Depth → [H, W, 4]
        fused = np.concatenate((im, depth), axis=2)
        
        # Update cache if enabled
        if self.augment or self.cache == "ram":
            self.ims[i] = fused
            
        return fused, hw0, hw

    @staticmethod
    def _process_depth_channel(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Clean, normalize, and confidence-weight a depth map before fusion.
        
        Args:
            depth: Raw depth array (H, W) or (H, W, 1) or (H, W, 3).
            target_hw: Target (height, width) to match RGB dimensions.
            
        Returns:
            np.ndarray: Processed depth [H, W, 1] in uint8 [0, 255].
            
        Processing Steps:
        1. Convert multi-channel to single channel (BGR→Gray if needed)
        2. Handle NaN/Inf (replace with 0)
        3. Resize to match RGB if needed (INTER_NEAREST preserves depth values)
        4. Median filter (5x5) → remove salt-and-pepper noise
        5. Gaussian filter (5x5) → smooth and fill small gaps
        6. Compute confidence map based on blur difference
        7. Percentile normalization (2%-98%) → adaptive range
        8. Confidence weighting → suppress unreliable regions
        9. Convert to uint8 [0, 255]
        
        📚 八股知识点: 为什么用INTER_NEAREST而非INTER_LINEAR？
        Q: 深度图resize为什么不用双线性插值？
        A: (1) 深度是离散测量值,不是连续信号
           (2) 线性插值会在边界产生错误的中间值
           (3) NEAREST保持原始测量,避免引入伪影
           (4) 特别是下采样时,NEAREST更鲁棒
           
        Example:
            深度边界: [1.0m, 1.0m, 5.0m, 5.0m]
            INTER_LINEAR: [1.0, 1.0, 3.0, 5.0, 5.0] ← 3.0m是错误的!
            INTER_NEAREST: [1.0, 1.0, 5.0, 5.0, 5.0] ← 正确!
        """
        # Step 1: Convert to single channel if needed
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[..., 0]  # Remove singleton dimension
            else:
                # Assume BGR depth visualization, convert to Gray
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        # Step 2: Handle NaN/Inf (common in depth sensors)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = depth.astype(np.float32, copy=False)

        # Step 3: Resize to target dimensions if needed
        if depth.shape != target_hw:
            depth = cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

        # Step 4-9: Denoise, normalize, weight
        valid_mask = depth > 0  # Identify valid depth measurements
        
        if valid_mask.any():
            # Step 4: Median filter (remove salt-and-pepper noise)
            # 📌 改进: 减小核尺寸3x3→保留小目标边缘 (vs ultralytics12的5x5)
            depth_blur = cv2.medianBlur(depth, 3)
            
            # Step 5: Gaussian filter (smooth and fill small gaps)
            depth_smooth = cv2.GaussianBlur(depth_blur, (5, 5), 0)
            
            # Step 6: Compute confidence (low diff = high confidence)
            diff = np.abs(depth_smooth - depth_blur)
            sigma = float(diff[valid_mask].std()) or 1.0
            confidence = np.clip(1.0 - diff / (3.0 * sigma + 1e-6), 0.0, 1.0)
            confidence *= valid_mask.astype(np.float32)

            # Step 7: Percentile normalization (adaptive to scene depth range)
            low, high = np.percentile(depth_blur[valid_mask], (2.0, 98.0))
            scale = max(high - low, 1e-6)
            depth_norm = np.clip((depth_blur - low) / scale, 0.0, 1.0)
            
            # Step 8: Confidence weighting (suppress unreliable regions)
            depth_weighted = depth_norm * confidence
        else:
            # No valid depth (all zeros/NaN)
            depth_weighted = np.zeros_like(depth, dtype=np.float32)

        # Step 9: Convert to uint8 for network input
        depth_uint8 = (depth_weighted * 255.0).astype(np.uint8)
        return depth_uint8[..., None]  # Add channel dimension [H, W, 1]


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (list[dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls, *s]
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [[*list(x), Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> list[tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
