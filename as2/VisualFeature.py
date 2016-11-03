from dircache import listdir
import numpy as np
import math
import re

def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]

def getVisualFeatures():
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
    #feature_mapping = {0: "Age", 2: "Angry", 3: "Happy",
    #               4: "MouthOpen", 5: "Surprised"}
    feature_mapping = {2: "Angry"}
    # feature_mapping = {134	:"right blow raise",135	:"left brow raise",136	:"brow squint"}

    results = []
    result = []
    counter = 0
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
                if (int(features[0]) >= end_frame):
                    break
                if math.isnan(feat_val):
                    continue
                else:
                    pool_feature.append(feat_val)
    
            if(sum(map(abs, pool_feature))!=0):
                pool_feature =[np.nanstd(pool_feature)]
            else:
                pool_feature = [0]
            if label == -1:
                class_negative_feature += pool_feature
            elif label == 1:
                class_positive_feature += pool_feature
            else:
                class_neutral_feature += pool_feature
            result+=pool_feature
        results.append(result)
        labels.append("SHORE-"+str(feature_mapping[i]))
    
    VISUAL_FEATURE = "../data/as1/features/VisualFeatures/CLM/"
    id_to_file_name = {}
    
    FEATURE = VISUAL_FEATURE
    
    for fn in listdir(FEATURE):
        id = fn[5:7]
        if id[1] == "(":
            id = str(id[0])
        id_to_file_name[int(str(id))] = fn
    
    # All feature mappings
    #feature_mapping = {0	:"frame",1	:"x_0",2	:"y_0",3	:"x_1",4	:"y_1",5	:"x_2",6	:"y_2",7	:"x_3",8	:"y_3",9	:"x_4",10	:"y_4",11	:"x_5",12	:"y_5",13	:"x_6",14	:"y_6",15	:"x_7",16	:"y_7",17	:"x_8",18	:"y_8",19	:"x_9",20	:"y_9",21	:"x_10",22	:"y_10",23	:"x_11",24	:"y_11",25	:"x_12",26	:"y_12",27	:"x_13",28	:"y_13",29	:"x_14",30	:"y_14",31	:"x_15",32	:"y_15",33	:"x_16",34	:"y_16",35	:"x_17",36	:"y_17",37	:"x_18",38	:"y_18",39	:"x_19",40	:"y_19",41	:"x_20",42	:"y_20",43	:"x_21",44	:"y_21",45	:"x_22",46	:"y_22",47	:"x_23",48	:"y_23",49	:"x_24",50	:"y_24",51	:"x_25",52	:"y_25",53	:"x_26",54	:"y_26",55	:"x_27",56	:"y_27",57	:"x_28",58	:"y_28",59	:"x_29",60	:"y_29",61	:"x_30",62	:"y_30",63	:"x_31",64	:"y_31",65	:"x_32",66	:"y_32",67	:"x_33",68	:"y_33",69	:"x_34",70	:"y_34",71	:"x_35",72	:"y_35",73	:"x_36",74	:"y_36",75	:"x_37",76	:"y_37",77	:"x_38",78	:"y_38",79	:"x_39",80	:"y_39",81	:"x_40",82	:"y_40",83	:"x_41",84	:"y_41",85	:"x_42",86	:"y_42",87	:"x_43",88	:"y_43",89	:"x_44",90	:"y_44",91	:"x_45",92	:"y_45",93	:"x_46",94	:"y_46",95	:"x_47",96	:"y_47",97	:"x_48",98	:"y_48",99	:"x_49",100	:"y_49",101	:"x_50",102	:"y_50",103	:"x_51",104	:"y_51",105	:"x_52",106	:"y_52",107	:"x_53",108	:"y_53",109	:"x_54",110	:"y_54",111	:"x_55",112	:"y_55",113	:"x_56",114	:"y_56",115	:"x_57",116	:"y_57",117	:"x_58",118	:"y_58",119	:"x_59",120	:"y_59",121	:"x_60",122	:"y_60",123	:"x_61",124	:"y_61",125	:"x_62",126	:"y_62",127	:"x_63",128	:"y_63",129	:"x_64",130	:"y_64",131	:"x_65",132	:"y_65",133	:"conf",134	:"right blow raise",135	:"left brow raise",136	:"brow squint"}
    feature_mapping = {134	:"right blow raise",135	:"left brow raise",136	:"brow squint"}
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
            if(video!=current_video):
                video = open(current_video, "r")
            start_frame = int(d_row[4])
            end_frame = int(d_row[5])
            label = int(d_row[6])
    
            pool_feature = []
            count = 0.0
            for frame in range(start_frame, end_frame, 1):
                features = video.readline().strip().split(" ")
                invalid = False
                for v in features:
                    if v.isdigit() and int(v)==0:
                        invalid = True
    
                if(invalid == True or int(features[0])==0):
                    continue
                if(i<134):
                    feat_val = float(features[i])
                elif i==134 :
                    feat_val = float(features[76]) - float(features[40])
                elif i==135:
                    feat_val = float(features[90]) - float(features[50])
                elif i == 136:
                    feat_val = float(features[45]) - float(features[43])
    
                if (int(features[0]) >= end_frame):
                    break
                #pool_feature+=feat_val # sum
                pool_feature.append(feat_val)
                count+=1
    
            if label == -1:
                class_negative_feature.append(np.std(pool_feature))
            elif label == 1:
                class_positive_feature.append(np.std(pool_feature))
            else:
                class_neutral_feature.append(np.std(pool_feature))
            result.append(np.std(pool_feature))
        results.append(result)
        labels.append("CLM-"+str(feature_mapping[i]))
    
    
    
    VISUAL_FEATURE = "../data/as1/features/VisualFeatures/OKAO/"
    id_to_file_name = {}
    FEATURE = VISUAL_FEATURE
    for fn in listdir(FEATURE):
        id = fn[5:7]
        if id[1] == "(":
            id = str(id[0])
        id_to_file_name[int(str(id))] = fn
    
    # All feature mappings
    #feature_mapping = {0	:"LTx",1	:"Lty",2	:"RBx",3	:"Rby",5	:"face_pose",6	:"RightEye0x",7	:"RightEye0y",9	:"RightEye1x",10	:"RightEye1y",12	:"RightEye2x",13	:"RightEye2y",15	:"RightEye3x",16	:"RightEye3y",18	:"RightEye4x",19	:"RightEye4y",21	:"RightEye5x",22	:"RightEye5y",24	:"RightEye6x",25	:"RightEye6y",27	:"RightEye7x",28	:"RightEye7y",30	:"LeftEye0x",31	:"LeftEye0y",33	:"LeftEye1x",34	:"LeftEye1y",36	:"LeftEye2x",37	:"LeftEye2y",39	:"LeftEye3x",40	:"LeftEye3y",42	:"LeftEye4x",43	:"LeftEye4y",45	:"LeftEye5x",46	:"LeftEye5y",48	:"LeftEye6x",49	:"LeftEye6y",51	:"LeftEye7x",52	:"LeftEye7y",54	:"mouth0x",55	:"mouth0y",57	:"mouth1x",58	:"mouth1y",60	:"mouth2x",61	:"mouth2y",63	:"mouth3x",64	:"mouth3y",66	:"mouth4x",67	:"mouth4y",69	:"mouth5x",70	:"mouth5y",72	:"mouth6x",73	:"mouth6y",75	:"mouth7x",76	:"mouth7y",78	:"mouth8x",79	:"mouth8y",81	:"mouth9x",82	:"mouth9y",84	:"mouth10x",85	:"mouth10y",87	:"mouth11x",88	:"mouth11y",90	:"mouth12x",91	:"mouth12y",93	:"mouth13x",94	:"mouth13y",96	:"mouth14x",97	:"mouth14y",99	:"mouth15x",100	:"mouth15y",102	:"mouth16x",103	:"mouth16y",105	:"mouth17x",106	:"mouth17y",108	:"mouth18x",109	:"mouth18y",111	:"mouth19x",112	:"mouth19y",114	:"mouth20x",115	:"mouth20y",117	:"mouth21x",118	:"mouth21y",120	:"face up down",121	:"face left right",122	:"face roll",124	:"gaze up down",125	:"gaze left right",127	:"left eye openness",129	:"right eye openness",131	:"mouth openness",133	:"smile level", 134 :"left_eye_size", 135:"right_eye_size"}
    feature_mapping = {127	:"left eye openness",129	:"right eye openness",133	:"smile level",120	:"face up down"}
    for i in feature_mapping.keys():
        video = ""
        result = []
        # Features in individual class
        class_negative_feature = []
        class_positive_feature = []
        class_neutral_feature = []
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
            else:
                class_neutral_feature.append(np.std(pool_feature))
            result.append(np.std(pool_feature))
        results.append(result);
        labels.append("OKAO-"+feature_mapping[i])
    return results,labels
