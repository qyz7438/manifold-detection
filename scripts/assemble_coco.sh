#!/bin/bash
set -e
cd /home/ps/lzz/RLimage/data/coco
if [ -f train2017.zip ]; then
    echo "train2017.zip already exists"
else
    cat train2017.zip.chunk_* > train2017.zip
    echo "assembled train2017.zip"
fi
unzip -q -t train2017.zip && echo "train zip OK"
rm -rf train2017
unzip -q train2017.zip && echo "extracted train2017"
