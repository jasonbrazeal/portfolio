#!/usr/bin/env bash

# set -euxo pipefail

# data_url='https://drive.google.com/file/d/1X6IyN2KS6cZiyzhpjfwMKtshn2tlPSo8/view'
# file_id=$(echo $data_url | sed -n 's/.*\/d\/\([^/]*\).*/\1/p')

echo 'Warning: this will clear the data and models directories and save new stuff there.'
echo 'Press Enter to continue or CTRL+C to cancel...'
read

curl -L "https://drive.usercontent.google.com/download?id=1X6IyN2KS6cZiyzhpjfwMKtshn2tlPSo8&export=download&confirm=t&uuid=b835a027-bb90-48d7-8a94-a8347152d629" -o 'data.tar.gz'

tar -xzf data.tar.gz
rm data.tar.gz

echo 'Data downloaded successfully'

##############################################################################

curl -L "https://drive.usercontent.google.com/uc?id=1r_FuJaMp_s3Sg9aziDn0UIf-cU9rRh2z&export=download" -o 'models.tar.gz'

tar -xzf models.tar.gz
rm models.tar.gz

echo 'Models downloaded successfully'
