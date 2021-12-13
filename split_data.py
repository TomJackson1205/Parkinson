#-*- coding: utf-8 -*-
import os
import csv

def split_csv(path,x,train_path,test_path):
    # If train.csv and vali.csv Delete if it exists
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)

    with open(path, 'r', newline='') as file:
        csvreader = csv.reader(file)
        j=1
        for i,row in enumerate(csvreader):
            if i>=x and i<x+3:
                csv_path = test_path
                # When this file does not exist, it is created
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    j += 1
                # Add to it when it exists
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    j += 1
            else:
                csv_path = train_path
                # When this file does not exist, it is created
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    j += 1
                # Add to it when it exists
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    j += 1
