# Extract patient specific folders from data root path and categories them according to the metadata xls file

## This code only needs to be ran ONCE but can take some time to run

import os
import shutil
import pandas as pd


anno_info = '/app/Alex/Dataset/Metadata/CCCCII_Data_Overview.xls' # [for UCL Cluster]
# anno_info = '/content/drive/My Drive/Thesis (Aladdin)/CCCCII_Data_Overview.xls' # [for Google Colab]
df=pd.read_excel(anno_info, sheet_name='Overview')

#this code takes in the file path of our source folder

data_folder = '/app/Alex/Dataset/covid_CCII/'  # [for Aladdin Cluster]
# data_folder = '/content/'  # [for Google Colab]


destination_folder= '/app/Alex/Images Destination Folder'  # [for Cluster]
# destination_folder= '/content/drive/My Drive/Thesis (Aladdin)/Images Destination Folder'  # [for Google Colab]

extension = '.png'

for index, row in df.iterrows():

    id= row['patient_id']


    label_name = row['Type']

    label= 1 if row['Type'] == 'COVID19' else 0

    source_folder = data_folder # + '/' + str(id)

    des_dir= destination_folder + '/' + str(label) + '/' +str(id)

    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for folders, _, filenames in os.walk(source_folder):
        print(filenames)
        for filename in filenames:
            if filename.endswith('{}'.format(extension)):
                shutil.copy(os.path.join(folders, filename), des_dir)