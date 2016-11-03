import csv
import numpy as np
from as2.VisualFeature import getVisualFeatures
from as2.AcousticFeatures import getAcousticFeatures

def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]

ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
data = np.genfromtxt(ANNOTATED_SENTI,
                     delimiter=",", skip_header=1, usecols=[0,1,2,9])

results = []
labels = []
result = []
ids = []
counter = 0
for d_row in data:
    counter = counter+1
    ids.append(str(d_row[0])+"_"+str(counter))
    result.append(d_row[3])
results.append(ids)
labels.append("Id")

visual_results, visual_labels = getVisualFeatures()
results = results + visual_results
labels = labels + visual_labels

#acoustic_result, acoustic_labels = getAcousticFeatures()
#results = results + acoustic_result
#labels = labels + acoustic_labels

results.append(result)
labels.append("Label")
results = list(map(list, zip(*results)))
results = [[]]+results
results[0] = labels
with open("output-feature-engineering-visual.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)

