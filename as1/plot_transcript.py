from matplotlib import pyplot as pl

from dircache import listdir
import numpy as np
import sys
from plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram
from readTrans_m import readTrans

def get_file_name(id_to_file_name, id):
    return id_to_file_name[str(int(id))]


ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"

TRANSCRIPT_FEATURE = "../data/as1/annotations/Transcriptions/"

id_to_file_name = {}

FEATURE_DIR = TRANSCRIPT_FEATURE

data = np.genfromtxt(ANNOTATED_SENTI,
                  delimiter=",", skip_header=1, usecols=[0,1,2,9])

for fn in listdir(FEATURE_DIR):
    vid_id = fn[5:7]
    if vid_id[1]=="(":
        vid_id=vid_id[0]
    id_to_file_name[vid_id]=fn 

for key in id_to_file_name:
    print readTrans(FEATURE_DIR + get_file_name(id_to_file_name, 1))[0]





