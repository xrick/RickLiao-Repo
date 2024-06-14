#!/bin/bash

read -p "the file folder to compress: " srcfolder

read -p "the target 7z path and file: " compressfilepath

read -p "the file clip size(for exampel:10m for 10MB): " clipsize

echo "compressing folder $srcfolder to $compressfilepath with clip size $clipsize"

7z -v$clipsize a $compressfilepath $srcfolder