#
# Tool set to update boundingdata table in sql DB
#
# Shu Peng
#
# ==============================================================================
import os
import numpy as np
import pyodbc

# SQL server
db_server = 'tncmodel.database.windows.net'
db_name = 'TNCModelTrainingDatabase'
login = 'tncadmin'
password = 'TNCAI4wildlife'
conn_str = "DRIVER={{SQL Server}};SERVER={};DATABASE={};UID={};PWD={}".format(db_server, db_name, login, password)

# BU (Beijing University) data set
data_set_name = 'BU'
train_img_width = 682
train_img_height = 512


def update_bounding_data(path):
    # Specifying the ODBC driver, server name, database, etc. directly
    cnxn = pyodbc.connect(conn_str)
    # Create a cursor from the connection
    cursor = cnxn.cursor()
    for root, dirs, files in os.walk(path):
        for img_name in files:
            if img_name[-4:] == '.JPG':
                base_name = img_name[:-4]
                bbox_name = base_name + '.bboxes.tsv'
                bboxes_file = os.path.join(root, bbox_name)
                if os.path.exists(bboxes_file):
                    bboxes = np.loadtxt(bboxes_file, np.float32)
                    print("{} bounding boxes in image {}".format(bboxes.ndim, base_name))
                    sql_insert = ''
                    if bboxes.ndim == 1:
                        top_x, top_y, bottom_x, bottom_y = bboxes
                        sql_insert += "INSERT INTO [dbo].[BoundingData] ([PictureName],[TopX],[TopY],[BottomX],[BottomY]," \
                                      "[Width],[Height]) VALUES ('{picture_name!s}',{topx:d},{topy:d},{botx:d},{boty:d}," \
                                      "{width:d},{height:d})\n".format(picture_name=base_name, topx=int(top_x),
                                                                       topy=int(top_y), botx=int(bottom_x),
                                                                       boty=int(bottom_y), width=int(train_img_width),
                                                                       height=int(train_img_height))
                    else:
                        for box in bboxes:
                            top_x, top_y, bottom_x, bottom_y = box
                            sql_insert += "INSERT INTO [dbo].[BoundingData] ([PictureName], [TopX], [TopY], [BottomX], [BottomY]," \
                                          "[Width],[Height]) VALUES ('{picture_name!s}',{topx:d},{topy:d},{botx:d},{boty:d}," \
                                          "{width:d},{height:d})\n".format(picture_name=base_name, topx=int(top_x),
                                                                           topy=int(top_y), botx=int(bottom_x),
                                                                           boty=int(bottom_y),
                                                                           width=int(train_img_width),
                                                                           height=int(train_img_height))
                    sql_delete = "DELETE FROM[dbo].[BoundingData] WHERE [PictureName]='{}'\n".format(base_name)
                    cursor.execute(sql_delete)
                    cursor.execute(sql_insert)
                    cnxn.commit()


if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(abs_path, "..\\DataSets", data_set_name)
    data_set_path = os.path.realpath(data_set_path)
    if os.path.exists(data_set_path):
        print(data_set_path)
    else:
        print("DataSets folder not found.")
    dir_names = [s for s in os.listdir(data_set_path) if
                 (os.path.isdir(os.path.join(data_set_path, s)) and (s[-7:] == '_output'))]

    for class_folder in dir_names:
        print("Update bounding data for images under ", os.path.join(data_set_path, class_folder))
        update_bounding_data(os.path.join(data_set_path, class_folder))
    print("Done.")
