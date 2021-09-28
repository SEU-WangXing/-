from __future__ import print_function
import moxing as mox
import zipfile
import os
import subprocess
from naie.context import Context


def loda_data():  # 加载数据并解压
    from naie.datasets import get_data_reference

    data_reference = get_data_reference(
        dataset="Default", dataset_entity="animal_classify_test"
    )
    file_paths = data_reference.get_file_paths()
    mox.file.copy(
        data_reference.get_file_paths()[0], "/cache/animal_classify_data_collect.zip"
    )
    rf = zipfile.ZipFile("/cache/animal_classify_data_collect.zip")
    rf.extractall("/cache/")
    rf.close()
    print(file_paths)


def main():
    config = "./config/resnet/resnet18_b32x8_imagenet.py"
    gpus = 8
    workdir = "./ckpt"
    cmd = "{}{}--work-dir{}".format(config, gpus, workdir)
    print(cmd)
    subproces.call("cd mmclassification_master && pip install -e.", shenll=True)
    subproces.call(
        "cd mmclassification_master && sh ./tools/dist_train.sh" + cmd, shenll=True
    )


if __name__ == "_main_":
    load_data()
    main()
    mox.file.copy(
        "/cache/user-job-dir/code/mmclassification/ckpt/latest.pth",
        os.payh.join(Context.get_model_path(), "latest,pth"),
    )
    os.path.abspath(".")
    print("path:", os.path.abspath("."))
    print("dir", os.listdir(os.path.abspath(".")))
