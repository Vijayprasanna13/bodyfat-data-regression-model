import csv
import os
import sys

"""
This function reads from the CSV file and returns the dataset
as an object list
:return: list of Bodyfat_data objects
:params: filename (string)
"""
def readCSV(filename):
	folder_path = os.path.dirname(os.path.abspath(__file__))
	#contains the data set after the reading of the csv file
	dataset  = []

	with open(folder_path+'/CSV/'+filename+'.csv','rb') as f:
		reader = csv.reader(f)
		for row in reader:
			 temp = row[0].split()
			 dataset.append(temp)

	#An object to hold the target Y and the attributes
	class DataPoint(object):
	 def __init__(self,target,features):
	  self.target = target
	  self.features = features

	#list of objects
	bodyfat_data = []

	#append features and target to object
	for data in dataset:
	 target = data[0]
	 del data[0]
	 features = []
	 for datum in data:
	  features.append( (datum.split(":"))[1] )
	 bodyfat_data.append(DataPoint(target,features))

	return bodyfat_data 
