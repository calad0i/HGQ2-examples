# HGQ2 Examples

Some examples of how to use HGQ2 for various tasks.

## JSC

There are two variants: OpenML and CERNBox.

For OpenML, data downloading is automatic.

For CERNBox, you need to download the data manually from [here](https://cernbox.cern.ch/s/jvFd5MoWhGs1l5v/download).

## Thin Gap Chamber

Dataset is available at [this](https://huggingface.co/datasets/Calad/fake-TGC) huggingface repo.


Please refer to the corresponding --help messages for more details on how to run each example. The `run_train.py` runs training and saves the checkpoints. The `run_test.py` runs evaluation and converts the models to rtl/hls projects.
