import csv
from datetime import datetime

# a function to write results to a CSV database for the 3 supported tasks
def write_results(base_path, db_loc,mode):
    with open(base_path, newline='', encoding='utf-8') as myfile:
        lines = list(csv.reader(myfile))
        last_row = lines[-1]

        col_3 = last_row[2]
        col_1 = last_row[1]
        
        today = datetime.today().strftime('%-d/%-m/%Y') 
        weights_path = last_row[5]
        config_path = last_row[6]
        if mode == "Seg":
            new_path_weights = "/data/Segmentation" + weights_path.split("/Segmentation", 1)[1]
            new_path_config = "/data/Segmentation" + config_path .split("/Segmentation", 1)[1]
            col_5 =  round(float(last_row[4]) * 100, 1)
        elif mode == "OD":
            new_path_weights = "/data" + weights_path.split("/ObjectDetection", 1)[1]
            new_path_config = "/data" + config_path .split("/ObjectDetection", 1)[1]
            col_5 =  round(float(last_row[4]), 4)
        elif mode == "Cls":
            new_path_weights = "/data/Classification" + weights_path.split("/Classification", 1)[1]
            new_path_config = "/data/Classification" + config_path .split("/Classification", 1)[1]
            col_5 =  round(float(last_row[4]), 2)

        new_row = [col_3, col_1, col_5, today,new_path_weights,new_path_config]

    with open(db_loc, 'a', newline='', encoding='utf-8') as modelfile:
        writer = csv.writer(modelfile)
        writer.writerow(new_row)
