#!/usr/bin/env bash

# set -euxo pipefail

# sharing_url='https://drive.google.com/file/d/1Tn75l-AsxV2ytAZQ5tf5YoCigPfQfxyE/view'
# file_id=$(echo $sharing_url | sed -n 's/.*\/d\/\([^/]*\).*/\1/p')

echo 'Warning: this will clear the data directory and save new data there.'
echo 'Press Enter to continue or CTRL+C to cancel...'
read

curl -L "https://drive.usercontent.google.com/download?id=1Tn75l-AsxV2ytAZQ5tf5YoCigPfQfxyE&export=download&confirm=t&uuid=142aacff-a725-473b-b227-617c89a4ec67" -o 'tech-manual-rag.tar.gz'
tar -xzf tech-manual-rag.tar.gz
rm tech-manual-rag.tar.gz

echo 'Data downloaded successfully'

