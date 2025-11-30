#!/bin/sh


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_path>"
    exit 1
fi

path="$1"
script_path=$(dirname $(realpath "$0"))

mkdir -p "$path"
cd "$path"

aria2c -s 32 -x 32 -o hls4ml_LHCjet_150p_train.tar.gz https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1
aria2c -s 32 -x 32 -o hls4ml_LHCjet_150p_val.tar.gz https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz?download=1

tar -xvzf hls4ml_LHCjet_150p_train.tar.gz
tar -xvzf hls4ml_LHCjet_150p_val.tar.gz

python3 "$script_path/prepare_data.py" -i ./train/ -o 150c-train.h5 -j 4
python3 "$script_path/prepare_data.py" -i ./val/ -o 150c-test.h5 -j 4

rm -rf ./train/
rm -rf ./val/
