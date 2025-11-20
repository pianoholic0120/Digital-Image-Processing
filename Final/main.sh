#!/bin/bash
cd /Users/arthurlin/Desktop/DIP/Final/dso/build
cmake ..
make -j4
bin/dso_dataset files=/Users/arthurlin/Desktop/DIP/Final/sequence_14/images \
                calib=/Users/arthurlin/Desktop/DIP/Final/sequence_14/camera.txt \
                gamma=/Users/arthurlin/Desktop/DIP/Final/sequence_14/pcalib.txt \
                vignette=/Users/arthurlin/Desktop/DIP/Final/sequence_14/vignette.png \
                preset=1 \
                mode=2