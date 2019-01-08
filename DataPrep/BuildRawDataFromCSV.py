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
import queue
import threading
import datetime

# Number of working thread
max_thread = 10

# All image source are saved under src_path
# <src_Path>\L-<location>\L-<location>-<camera>\*.JPG
src_path = 'E:\\BeijingUniversity'

# By default we copy the source image into 'Raw' DataSet
destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\DataSets\\Raw")
# Whether we should create location folder under destination path?
# <destination_path>\\Animal\\<location>\\image1.jpg
#                                       \\image2.jpg
create_location_sub_folder = False

# Size of the training images
width = 682
height = 512


def transform(x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        return 0, 0
    return int(x * width / w), int(y * height / h)


task_queue = queue.Queue()
threads = []
results_lock = threading.Lock()
missing_file_list = []
skipped_file_list = []


def worker():
    global missing_file_list, skipped_file_list, results_lock, task_queue
    while True:
        # Each item is a DataFrame object
        item = task_queue.get()
        missing_files = []
        skipped_files = []
        if item is None:
            break
        # Working on all item in the thread.
        class_name = item.iloc[0]['Name'].replace('\n', '', 10).replace(' ', '_', 10)
        print("Worker thread is processing {!s}...".format(class_name))
        if not os.path.exists(os.path.join(destination_path, class_name)):
            os.makedirs(os.path.join(destination_path, class_name))
        print("Total images: {:d}".format(item.shape[0]))

        for it, row in item.iterrows():
            # skip the picture that has no roi data.
            if row['BottomX'] == 0 and row['BottomY'] == 0 and class_name != 'Empty':
                skipped_files.append(row['FileName'])
                continue
            l, location, camera, sequence = row['FileName'].split('-')
            src_file = os.path.join(src_path, "L-" + location, "L-" + location + "-" + camera, row['FileName'] + '.JPG')
            if not os.path.exists(src_file):
                missing_files.append(row['FileName'])
                continue
            if create_location_sub_folder:
                dst_folder = os.path.join(destination_path, class_name, "L-" + location + "-" + camera)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
            else:
                dst_folder = os.path.join(destination_path, class_name)
            dst_file = os.path.join(dst_folder, row['FileName'] + '.JPG')
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            shutil.copyfile(src_file, dst_file)

            # create .bboxes.labels.tsv file
            label_tsv_file = row['FileName'] + '.bboxes.labels.tsv'
            with open(os.path.join(dst_folder, label_tsv_file), 'a', encoding='utf-8') as f:
                line = class_name + '\n'
                f.write(line)
            # create .bboxes.tsv file
            tsv_file = row['FileName'] + '.bboxes.tsv'
            with open(os.path.join(dst_folder, tsv_file), 'a') as f:
                line = "{top_x:d}\t{top_y:d}\t{bot_x:d}\t{bot_y:d}\n"
                top_x, top_y = transform(row['TopX'], row['TopY'], row['Width'], row['Height'])
                bot_x, bot_y = transform(row['BottomX'], row['BottomY'], row['Width'], row['Height'])
                f.write(line.format(top_x=top_x, top_y=top_y, bot_x=bot_x, bot_y=bot_y))
        results_lock.acquire(blocking=True)
        missing_file_list += missing_files
        skipped_file_list += skipped_files
        results_lock.release()
        task_queue.task_done()


def main():
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
    df.drop_duplicates(subset=['FileName'], inplace=True)
    unique_species = df['Name'].unique()
    print("{} species found in csv file.".format(len(unique_species)))

    start_time = datetime.datetime.now()

    for i in range(max_thread):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for species in unique_species:
        species_df = df[df['Name'] == species]
        task_queue.put(species_df)

    task_queue.join()

    for i in range(max_thread):
        task_queue.put(None)

    for t in threads:
        t.join()

    time_spent = datetime.datetime.now() - start_time

    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("{:d} image(s) were missing.".format(len(missing_file_list)))
    for filename in missing_file_list:
        print(filename)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("{:d} image(s) were skipped because the roi data is empty.".format(len(skipped_file_list)))
    for filename in skipped_file_list:
        print(filename)
    print("Done! Total time spent: {!s}".format(time_spent))


if __name__ == '__main__':
    main()
