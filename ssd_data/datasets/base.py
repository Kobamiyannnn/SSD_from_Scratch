import torch
from torch.utils.data import Dataset
import numpy as np
import abc

from .._utils import _check_ins, _contain_ignore
from ..target_transforms import Ignore

"""
ref > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample

"""


class _DatasetBase(Dataset):
    @property
    @abc.abstractmethod
    def class_nums(self):
        pass

    @property
    @abc.abstractmethod
    def class_labels(self):
        pass


class ObjectDetectionDatasetBase(_DatasetBase):
    def __init__(
        self, ignore=None, transform=None, target_transform=None, augmentation=None
    ):
        """
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation: instance of augmentations
        """
        self.ignore = _check_ins("ignore", ignore, Ignore, allow_none=True)
        self.transform = transform
        self.target_transform = _contain_ignore(target_transform)
        self.augmentation = augmentation

    @property
    @abc.abstractmethod
    def class_nums(self):
        pass

    @property
    @abc.abstractmethod
    def class_labels(self):
        pass

    @abc.abstractmethod
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(Tensor)
        """
        raise NotImplementedError('"_get_image" must be overridden.')

    @abc.abstractmethod
    def _get_target(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated])
        """
        raise NotImplementedError('"_get_target" must be overridden.')

    def __getitem__(self, index):
        """
        与えられた`index`に対する画像とバウンディングボックスを取得する

        Parameters
        ----------
        index : int
            _description_

        Returns
        -------
        img : Tensor or ndarray
        targets : Tensor or ndarray of bboxes and labels [box, label]
            = [xmin, ymin, xmamx, ymax, label index(or relu_one-hotted label)]
            or
            = [cx, cy, w, h, label index(or relu_one-hotted label)]
        """
        img = self._get_image(index)
        targets = self._get_target(index)
        if len(targets) >= 3:
            bboxes, linds, flags = targets[:3]
            args = targets[3:]
        else:
            raise ValueError(
                "ValueError: not enough values to unpack (expedtec more than 3, got {})".format(
                    len(targets)
                )
            )
        img, bboxes, linds, flags, args = self.apply_transform(
            img, bboxes, linds, flags, *args
        )

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(linds, torch.Tensor):
            if linds.ndim == 1:
                linds = linds.unsqueeze(1)
            targets = torch.cat((bboxes, linds), dim=1)
        else:
            if linds.ndim == 1:
                linds = linds[:, np.newaxis]
            targets = np.concatenate((bboxes, linds), axis=1)

        return img, targets
