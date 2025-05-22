#!/usr/bin/env bash

set -euxo pipefail

echo 'Warning: this will clear/create the output directory and save new data there.'
echo 'Press Enter to continue or CTRL+C to cancel...'
read

curl -L "https://drive.usercontent.google.com/download?id=1oso6i3bI4biCT1DCGiMV0D9jy4sXQIqo&export=download&confirm=t&uuid=e3dde947-e74c-4be3-900d-1d39d3ddd1f6" -o 'output.tar.gz'
tar -xzf output.tar.gz
rm output.tar.gz

echo 'Output downloaded successfully'

