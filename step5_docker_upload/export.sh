#!/usr/bin/env bash

./build.sh

docker save cldetection_alg_2024 | gzip -c > CLdetection_Alg_2024.tar.gz
