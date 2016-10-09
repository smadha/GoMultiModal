from matplotlib import pyplot as pl

from dircache import listdir
import numpy as np
import math
import re
import sys
from plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram


def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]


ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
VISUAL_FEATURE = "../data/as1/features/VisualFeatures/SHORE/"
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
feature_mapping = {0: "Age", 2: "Angry", 3: "Happy",
                   4: "MouthOpen", 5: "Surprised"}
# feature_mapping = {134	:"right blow raise",135	:"left brow raise",136	:"brow squint"}


for i in feature_mapping.keys():
    # video feature file to be read
    video = ""

    # Features in individual class
    class_negative_feature = []
    class_positive_feature = []

    for d_row in data:
        current_video = FEATURE + get_file_name(id_to_file_name, d_row[0])
        if (video != current_video):
            video = open(current_video, "r")
        start_frame = int(d_row[4])
        end_frame = int(d_row[5])
        label = int(d_row[6])

        pool_feature = []
        for frame in range(start_frame, end_frame, 1):
            line = video.readline()
            features = line.strip().split(" ")
            if (bool(re.compile('Rating {(( \|--([\w]+) = ([\d.]+) )+)}').search(line))==False):
                continue;
            m = re.search('Rating {(( \|--([\w]+) = ([\d.]+) )+)}', line)
            interesting_part = m.groups()[0]
            tokens = interesting_part.split('|--')
            tokens_dict = {}
            for token in tokens:
                elements = token.split("=")
                if (len(elements) == 2):
                    tokens_dict[elements[0].strip()]=elements[1].strip();
            if (int(features[0]) == 0):
                continue
            if feature_mapping[i] in tokens_dict:
                feat_val = float(tokens_dict[feature_mapping[i]])
            else:
                continue
            if math.isnan(feat_val):
                continue
            else:
                pool_feature.append(feat_val)

            if (int(features[0]) >= end_frame):
                break
        if(sum(map(abs, pool_feature))!=0):
            pool_feature =[np.nanstd(pool_feature)]
        else:
            pool_feature = [0]
        # sum
                # pool_feature = max(pool_feature, feat_val)
        if label == -1:
            class_negative_feature += pool_feature
        elif label == 1:
            class_positive_feature += pool_feature

    axis = [0,
            100000,
            min(min(class_negative_feature), min(class_positive_feature)) - 1,
            max(max(class_negative_feature), max(class_positive_feature)) + 1]

    #     plot_scatter(class_1_feature, "class-1-"+feat_id_to_name[i],axis)
    #     plot_scatter(class1_feature, "class1-"+feat_id_to_name[i],axis)
    #
    #     plot_histogram(class_1_feature, "class-1-"+feat_id_to_name[i])
    # plot_histogram(class1_feature, "class1-"+feat_id_to_name[i])
    #print class_positive_feature
    #print class_negative_feature
    data_to_plot = [class_positive_feature, class_negative_feature]
    plot_box(data_to_plot, "SHORE", feature_mapping[i], ["class 1", "class -1"])
    # plot_xy_histogram(class_positive_feature, class_negative_feature, feature_mapping[i], ["class1-", "class-1-"])


