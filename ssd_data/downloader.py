import glob
import logging
import os
import shutil
import tarfile
import zipfile

import pycurl

from ._utils import DATA_ROOT

choices = ["coco2014_trainval", "coco2017_trainval"]
__all__ = choices


class _Downloader:
    def __init__(self, url, compress_ext="tar"):
        """
        Initializes the downloader with the given URL and compression extension.

        Args:
            url (str): The URL to download the data from.
            compress_ext (str, optional): The compression extension to use.
                                          Must be either 'tar' or 'zip'. Defaults to 'tar'.

        Raises:
            ValueError: If the provided compression extension is not 'tar' or 'zip'.
        """
        self.url = url

        _compress_exts = ["tar", "zip"]
        if compress_ext not in _compress_exts:
            raise ValueError(
                "Invalid argument (got {}): 適切なアーカイブ形式を選択してください ({})。".format(
                    compress_ext, _compress_exts
                )
            )
        self.compress_ext = compress_ext

    def run(self, out_base_dir, dirname, remove_comp_file=True):
        """
        Downloads and extracts a dataset from a specified URL.
        Args:
            out_base_dir (str): The base directory where the dataset will be saved.
            dirname (str): The name of the directory where the dataset will be extracted.
            remove_comp_file (bool, optional): Whether to remove the compressed file after extraction. Defaults to True.
        Raises:
            AssertionError: If the compression extension is not supported.
        Returns:
            None
        """
        out_dir = os.path.join(out_base_dir, dirname)

        if len(glob.glob(os.path.join(out_dir, "*"))) > 0:
            logging.warning(
                'データセットはすでにダウンロードされています。再度ダウンロードする場合は、ディレクトリ ("{}") を削除してください。'.format(
                    out_base_dir
                )
            )
            return

        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, self.url)
        curl.setopt(pycurl.FOLLOWLOCATION, True)  # リダイレクトの許可
        curl.setopt(pycurl.NOPROGRESS, False)  # 進捗の表示

        os.makedirs(out_dir, exist_ok=True)

        # データセットの圧縮ファイルのパス
        dstpath = os.path.join(out_base_dir, f"{dirname}.{self.compress_ext}")

        with open(dstpath, "wb") as f:
            curl.setopt(pycurl.WRITEFUNCTION, f.write)
            curl.perform()

        curl.close()

        # 解凍
        if self.compress_ext == "tar":
            with tarfile.open(dstpath) as tf:
                tf.extractall(out_dir)
        elif self.compress_ext == "zip":
            with zipfile.open(dstpath) as zf:
                zf.extractall(out_dir)
        else:
            assert False, "不明なエラー"

        if remove_comp_file:
            os.remove(dstpath)


def _concat_trainval_images(base_dir, src_dirs=("train", "val"), dst_dir="trainval"):
    """
    Concatenate images from source directories into a destination directory.

    This function moves all images from the specified source directories
    (default: "train" and "val") within the base directory to a destination
    directory (default: "trainval"). After moving the images, it removes the
    source directories if they are empty.

    Args:
        base_dir (str): The base directory containing the source directories.
        src_dirs (tuple, optional): A tuple of source directory names to move
            images from. Defaults to ("train", "val").
        dst_dir (str, optional): The name of the destination directory to move
            images to. Defaults to "trainval".

    Raises:
        AssertionError: If the images could not be moved to the destination
            directory.
    """
    src_paths = []
    for src_name in src_dirs:
        src_paths.extend(
            glob.glob(os.path.join(DATA_ROOT + base_dir, src_name, "images", "*"))
        )

    dst_path = os.path.join(DATA_ROOT + base_dir, dst_dir, "images")

    os.makedirs(dst_path, exist_ok=True)

    for src_path in src_paths:
        shutil.move(src_path, dst_path)

    if len(glob.glob(os.path.join(dst_path, "*"))) > 0:
        # remove source
        for src_name in src_dirs:
            shutil.rmtree(os.path.join(DATA_ROOT + base_dir, src_name))
        else:
            raise AssertionError("ファイルを移動できません。")


def coco2014_trainval():
    logging.info("COCO 2014 データセットをダウンロードしています。")

    # アノテーションファイルのダウンロード
    trainval_downloader = _Downloader(
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip", "zip"
    )
    trainval_downloader.run(DATA_ROOT + "/coco/coco2014", "trainval", remove_comp_file=True)

    # 画像データのダウンロード
    train_downloader = _Downloader("http://images.cocodataset.org/zips/train2014.zip", "zip")
    train_downloader.run(DATA_ROOT + "/coco/coco2014/train", "images", remove_comp_file=True)

    val_downloader = _Downloader("http://images.cocodataset.org/zips/val2014.zip", "zip")
    val_downloader.run(DATA_ROOT + "/coco/coco2014/val", "images", remove_comp_file=True)

    _concat_trainval_images("/coco/coco2014", srcdirs=("train", "val"), dstdir="trainval")

    logging.info("COCO 2014 データセットのダウンロードが完了しました。")


def coco2017_trainval():
    logging.info("COCO 2017 データセットをダウンロードしています。")

    # アノテーションファイルのダウンロード
    trainval_downloader = _Downloader(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "zip"
    )
    trainval_downloader.run(DATA_ROOT + "/coco/coco2017", "trainval", remove_comp_file=True)

    # 画像データのダウンロード
    train_downloader = _Downloader("http://images.cocodataset.org/zips/train2017.zip", "zip")
    train_downloader.run(DATA_ROOT + "/coco/coco2017/train", "images", remove_comp_file=True)

    val_downloader = _Downloader("http://images.cocodataset.org/zips/val2017.zip", "zip")
    val_downloader.run(DATA_ROOT + "/coco/coco2017/val", "images", remove_comp_file=True)

    _concat_trainval_images("/coco/coco2017", srcdirs=("train", "val"), dstdir="trainval")

    logging.info("COCO 2017 データセットのダウンロードが完了しました。")
