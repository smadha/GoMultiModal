from matplotlib import pyplot as pl

from dircache import listdir
import numpy as np
import sys
from plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram


def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]


ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
VISUAL_FEATURE = "../data/as1/features/VisualFeatures/OKAO/"
id_to_file_name = {}

FEATURE = VISUAL_FEATURE

data = np.genfromtxt(ANNOTATED_SENTI,
                     delimiter=",", skip_header=1, usecols=[0, 1, 2, 3, 4, 5, 9])

for fn in listdir(FEATURE):
    id = fn[5:7]
    if id[1] == "(":
        id = str(id[0])
    id_to_file_name[int(str(id))] = fn

# All feature mappings
feature_mapping = {0	:"LTx",1	:"Lty",2	:"RBx",3	:"Rby",5	:"face_pose",6	:"RightEye0x",7	:"RightEye0y",9	:"RightEye1x",10	:"RightEye1y",12	:"RightEye2x",13	:"RightEye2y",15	:"RightEye3x",16	:"RightEye3y",18	:"RightEye4x",19	:"RightEye4y",21	:"RightEye5x",22	:"RightEye5y",24	:"RightEye6x",25	:"RightEye6y",27	:"RightEye7x",28	:"RightEye7y",30	:"LeftEye0x",31	:"LeftEye0y",33	:"LeftEye1x",34	:"LeftEye1y",36	:"LeftEye2x",37	:"LeftEye2y",39	:"LeftEye3x",40	:"LeftEye3y",42	:"LeftEye4x",43	:"LeftEye4y",45	:"LeftEye5x",46	:"LeftEye5y",48	:"LeftEye6x",49	:"LeftEye6y",51	:"LeftEye7x",52	:"LeftEye7y",54	:"mouth0x",55	:"mouth0y",57	:"mouth1x",58	:"mouth1y",60	:"mouth2x",61	:"mouth2y",63	:"mouth3x",64	:"mouth3y",66	:"mouth4x",67	:"mouth4y",69	:"mouth5x",70	:"mouth5y",72	:"mouth6x",73	:"mouth6y",75	:"mouth7x",76	:"mouth7y",78	:"mouth8x",79	:"mouth8y",81	:"mouth9x",82	:"mouth9y",84	:"mouth10x",85	:"mouth10y",87	:"mouth11x",88	:"mouth11y",90	:"mouth12x",91	:"mouth12y",93	:"mouth13x",94	:"mouth13y",96	:"mouth14x",97	:"mouth14y",99	:"mouth15x",100	:"mouth15y",102	:"mouth16x",103	:"mouth16y",105	:"mouth17x",106	:"mouth17y",108	:"mouth18x",109	:"mouth18y",111	:"mouth19x",112	:"mouth19y",114	:"mouth20x",115	:"mouth20y",117	:"mouth21x",118	:"mouth21y",120	:"face up down",121	:"face left right",122	:"face roll",124	:"gaze up down",125	:"gaze left right",127	:"left eye openness",129	:"right eye openness",131	:"mouth openness",133	:"smile level", 134 :"left_eye_size", 135:"right_eye_size"}


for i in feature_mapping.keys():
    # video feature file to be read
    video = ""

    # Features in individual class
    class_negative_feature = []
    class_positive_feature = []

    for d_row in data:
        current_video = FEATURE + get_file_name(id_to_file_name, d_row[0])
        if(video!=current_video):
            video = open(current_video, "r")
        start_frame = int(d_row[4])
        end_frame = int(d_row[5])
        label = int(d_row[6])

        pool_feature = []
        count = 0;
        for frame in range(start_frame, end_frame, 1):
            features = video.readline().strip().split(" ")
            if(len(features)==1):
                continue
            if(i<134):
                feat_val = float(features[i])
            elif i==134 :
                feat_val = float(features[40]) - float(features[37])
            elif i==135:
                feat_val = float(features[16]) - float(features[13])
            #             pool_feature+=feat_val # sum
            pool_feature.append(feat_val)
            count+=1

        if label == -1:
            class_negative_feature.append(np.std(pool_feature))
        elif label == 1:
            class_positive_feature.append(np.std(pool_feature))

    axis = [0,
            100000,
            min(min(class_negative_feature), min(class_positive_feature)) - 1,
            max(max(class_negative_feature), max(class_positive_feature)) + 1]

    #     plot_scatter(class_1_feature, "class-1-"+feat_id_to_name[i],axis)
    #     plot_scatter(class1_feature, "class1-"+feat_id_to_name[i],axis)
    #
    #     plot_histogram(class_1_feature, "class-1-"+feat_id_to_name[i])
    #     plot_histogram(class1_feature, "class1-"+feat_id_to_name[i])

    data_to_plot = [class_positive_feature,class_negative_feature]
    plot_box(data_to_plot,"OKAO", feature_mapping[i],["image/class 1","image/class -1"])
    #plot_xy_histogram(class1_feature, class_1_feature, feat_id_to_name[i], ["image/class1-", "image/class-1-"])


