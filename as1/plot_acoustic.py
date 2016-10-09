from matplotlib import pyplot as pl

from dircache import listdir
import numpy as np
import sys
from plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram

def get_file_name(id_to_file_name, id):
    return id_to_file_name[str(int(id))]

ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
ACOUSTIC_FEATURE = "../data/as1/features/AcousticFeatures/"
# dict to store video id to video name
# 1 -> video1(00h00m27s-00h01m01s).wav 
id_to_file_name = {}

FEATURE_DIR = ACOUSTIC_FEATURE

data = np.genfromtxt(ANNOTATED_SENTI,
                  delimiter=",", skip_header=1, usecols=[0,1,2,9])

for fn in listdir(FEATURE_DIR):
    id = fn[5:7]
    if id[1]=="(":
        id=id[0]
    id_to_file_name[id]=fn

# All acoustic
feat_id_to_name = {4:"energy-dB",5:"energy-slope",3:"frequency",2:"NAQ",1:"PeakSlope",6:"stationarity",0:"Voiced-Unvoiced"}


              
for i in feat_id_to_name.keys():
    #video feature file to be read 
    vid_last = open(FEATURE_DIR + get_file_name(id_to_file_name, data[0][0]),"r")
    print "Read - ", vid_last.name
    
    # Features in individual class
    class_1_feature = []
    class1_feature = []
    
    for d_row in data:
        vid_new = FEATURE_DIR + get_file_name(id_to_file_name,d_row[0])
        start_ms = int(d_row[1]*100)*10
        end_ms = int(d_row[2]*100)*10
        label = int(d_row[3])
        
        if (vid_new != vid_last.name):
            vid_last = open(vid_new ,"r")
            print "Read - ",vid_new
            
        for obs_range in range(start_ms,end_ms,10):
            features = vid_last.readline().strip().split(",")
            feat_val = float(features[i])
            
            if str(feat_val)[-3:] == "inf" or str(feat_val) == "nan"  or (abs(feat_val)>18 and i==1)  :
                print feat_val,label
                continue
             
            if label == -1:
                class_1_feature.append(feat_val )
            elif label == 1:
                class1_feature.append(feat_val )
                
    axis = [0, 100000, min(min(class_1_feature),min(class1_feature))-1, max(max(class_1_feature),max(class1_feature))+1]
            
    print len(class_1_feature)
    print len(class1_feature)
    
#     plot_scatter(class_1_feature, "class-1-"+feat_id_to_name[i],axis)
#     plot_scatter(class1_feature, "class1-"+feat_id_to_name[i],axis)
#     
#     plot_histogram(class_1_feature, "class-1-"+feat_id_to_name[i])
#     plot_histogram(class1_feature, "class1-"+feat_id_to_name[i])
    
#     data_to_plot = [class1_feature,class_1_feature]
#     plot_box(data_to_plot, feat_id_to_name[i],["class 1","class -1"])
    plot_xy_histogram(class1_feature,class_1_feature, feat_id_to_name[i],["class1-","class-1-"])
    
    
    