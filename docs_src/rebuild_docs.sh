#!/bin/bash

rm -rf ./source/s3prl.*
rm -rf ./source/_autosummary

make clean html

cp -r build/html/* ../docs/

