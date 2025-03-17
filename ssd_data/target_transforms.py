import logging

import numpy as np
import torch

from ._utils import _check_ins, _one_hot_encode


class Compose(object):
    def __init__(self, target_transforms):
        self.target_transforms = target_transforms

    def __call__(self, bboxes, labels, flags, *args):
        for t in self.target_transforms:
            bboxes, labels, flags, args = t(bboxes, labels, flags, *args)
        return bboxes, labels, flags, args

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.target_transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Ignore(object):
    supported_key = ["difficult", "truncated", "occluded", "iscrowd"]

    def __init__(self, **kwargs):
        """
        :param kwargs: if true, specific keyword will be ignored.
        """
        self.ignore_key = []
        for key, val in kwargs.items():
            val = _check_ins(key, val, bool)
            if not val:
                logging.warning("No meaning: {}=False".format(key))
            else:
                self.ignore_key += [key]
        else:
            logging.warning("Unsupported arguments: {}".format(key))

    def __call__(self, bboxes, labels, flags, *args):
        ret_bboxes = []
        ret_labels = []
        ret_flags = []

        for bbox, label, flag in zip(bboxes, labels, flags):
            flag_keys = list(flag.keys())
            ig_flag = [
                flag[ig_key] if ig_key in flag_keys else False for ig_key in self.ignore_key
            ]
            if any(ig_flag):
                continue

            # normalize
            # bbox = [xmin, ymin, xmax, ymax]
            ret_bboxes += [bbox]
            ret_labels += [label]
            ret_flags += [flag]

        ret_bboxes = np.array(ret_bboxes, dtype=np.float32)
        ret_labels = np.array(ret_labels, dtype=np.float32)

        return ret_bboxes, ret_labels, ret_flags, args
