#
# Tool set for CNTK Faster-RCNN training
#
# Shu Peng
#
# ==============================================================================
# Create the csv from DB query -
"""
SELECT [FileName]
      ,[Format]
      ,[TopX]
      ,[TopY]
      ,[BottomX]
      ,[BottomY]
      ,[Width]
      ,[Height]
      ,[Name]
      ,[ChnName]
      ,[SpeciesID]
  FROM [TNC].[dbo].[View_Bounding_JPG]
  Where TopX is not NULL and TopY is not NULL
"""

import os
import pandas as pd
import shutil
import sys

src_path = 'E:\\TNC_RawData\\BeijingUniversity'
destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\DataSets\\Raw")

# Size of the training images
width = 682
height = 512

# Whether we should create location folder under destination path?
# <destination_path>\\Animal\\<location>\\image1.jpg
#                                       \\image2.jpg
create_location_sub_folder = False


def transform(x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        return 0, 0
    return int(x * width / w), int(y * height / h)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print("Please provide a bounding data file (*.csv) ")
        print("Usage:")
        print("    python {} <bounding.csv>".format(os.path.basename(args[0])))
        exit()
    csv_path = args[1]
    if not os.path.exists(csv_path):
        print("Error: file not found -", csv_path)
        exit()
    df = pd.read_csv(csv_path)
    if 'FileName' not in df.columns or 'Name' not in df.columns:
        print("Please make sure {} contains both FileName and Name columns".format(csv_path))
        exit()
    unique_species = df['Name'].unique()
    print("{} species found in csv file.".format(len(unique_species)))
    for species in unique_species:
        class_name = species.replace('\n', '', 10).replace(' ', '_', 10)
        print("Processing {}...".format(species))
        species_df = df[df['Name'] == species]
        if not os.path.exists(os.path.join(destination_path, class_name)):
            os.makedirs(os.path.join(destination_path, class_name))
        print("Total images: {:d}".format(species_df.shape[0]))
        for item, row in species_df.iterrows():
            l, location, camera, squence = row['FileName'].split('-')
            src_file = os.path.join(src_path, "L-" + location, "L-" + location + "-" + camera, row['FileName'] + '.JPG')
            if create_location_sub_folder:
                dest_folder = os.path.join( destination_path, class_name, "L-" + location + "-" + camera)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
            else:
                dest_folder = os.path.join(destination_path, class_name)
            dest_file = os.path.join(dest_folder, row['FileName'] + '.JPG')
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            # print("{!s} => {!s}".format(src_file, dest_file))
            if not os.path.exists(src_file):
                print("dir " + src_file)
            shutil.copyfile(src_file, dest_file)
            # create .bboxes.labels.tsv file
            label_tsv_file = row['FileName'] + '.bboxes.labels.tsv'
            with open(os.path.join(dest_folder, label_tsv_file), 'w', encoding='utf-8') as f:
                line = class_name + '\n'
                f.write(line)
                f.flush()
            # create .bboxes.tsv file
            tsv_file = row['FileName'] + '.bboxes.tsv'
            with open(os.path.join(dest_folder, tsv_file), 'w') as f:
                line = "{top_x:d}\t{top_y:d}\t{bot_x:d}\t{bot_y:d}\n"
                top_x, top_y = transform(row['TopX'], row['TopY'], row['Width'], row['Height'])
                bot_x, bot_y = transform(row['BottomX'], row['BottomY'], row['Width'], row['Height'])
                f.write(line.format(top_x=top_x, top_y=top_y, bot_x=bot_x, bot_y=bot_y))
                f.flush()
    print("Done!")
