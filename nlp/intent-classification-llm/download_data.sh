#!/usr/bin/env bash

# set -euxo pipefail

echo 'Warning: this will clear the data directory and save new stuff there.'
echo 'Press Enter to continue or CTRL+C to cancel...'
read

curl -L "https://drive.usercontent.google.com/download?id=1QTQEcMrhld2kdsxVPTlxzfMvRqmBfVci&export=download&confirm=t&uuid=48604212-0ac5-4774-a7bf-df2d8242fb23" -o 'data.tar.gz'

tar -xzf data.tar.gz
rm data.tar.gz

echo 'Data downloaded successfully'
