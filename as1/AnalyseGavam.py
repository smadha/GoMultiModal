from matplotlib import pyplot as pl

from dircache import listdir
import numpy as np
import csv
import sys
from plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram


def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]


ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
VISUAL_FEATURE = "../data/as1/features/VisualFeatures/GAVAM/"
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
feature_mapping = {0: "frame", 1: "time occurence", 2: "pose est", 3: "X displacement",
                   4: "Y displacement", 5: "Z displacement", 6: "X angular disp",
                   7: "Y angular disp", 8: "Z angular disp", 9: "conf"}
# feature_mapping = {134	:"right blow raise",135	:"left brow raise",136	:"brow squint"}

results = []
labels = []
for i in feature_mapping.keys():
    # video feature file to be read
    video = ""

    # Features in individual class
    class_negative_feature = []
    class_positive_feature = []
    class_neutral_feature = []
    result = []
    for d_row in data:
        current_video = FEATURE + get_file_name(id_to_file_name, d_row[0])
        if (video != current_video):
            video = open(current_video, "r")
        start_frame = int(d_row[4])
        end_frame = int(d_row[5])
        label = int(d_row[6])

        pool_feature = []
        for frame in range(start_frame, end_frame, 1):
            features = video.readline().strip().split(" ")
            invalid = False
            for v in features:
                if v.isdigit() and int(v) == 0:
                    invalid = True

            if (invalid == True or int(features[0]) == 0):
                continue
            feat_val = float(features[i])
            if (int(features[0]) >= end_frame):
                break

            # pool_feature+=feat_val # sum
            pool_feature.append(feat_val)

        if label == -1:
            class_negative_feature.append(np.std(pool_feature))
        elif label == 1:
            class_positive_feature.append(np.std(pool_feature))
        else:
            class_neutral_feature.append(np.std(pool_feature))
        result.append(np.std(pool_feature))
    results.append(result)
    axis = [0,
            100000,
            min(min(class_negative_feature), min(class_positive_feature)) - 1,
            max(max(class_negative_feature), max(class_positive_feature)) + 1]

    #     plot_scatter(class_1_feature, "class-1-"+feat_id_to_name[i],axis)
    #     plot_scatter(class1_feature, "class1-"+feat_id_to_name[i],axis)
    #
    #     plot_histogram(class_1_feature, "class-1-"+feat_id_to_name[i])
    # plot_histogram(class1_feature, "class1-"+feat_id_to_name[i])

    data_to_plot = [class_positive_feature, class_negative_feature, class_neutral_feature]
    plot_box(data_to_plot, "GAVAM", feature_mapping[i], ["class 1", "class -1", "class 0"])
    # plot_xy_histogram(class_positive_feature, class_negative_feature, feature_mapping[i], ["class1-", "class-1-"])
    labels.append("GAVAM-"+str(feature_mapping[i]))
results = list(map(list, zip(*results)))

results = [[]]+results
results[0] = labels
with open("output-GAVAM.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)


