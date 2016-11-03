from dircache import listdir
import numpy as np
import string
from as1.readTrans_m import readTrans
from as1.obscene_word import profanity_count

stop_char = string.punctuation + '1234567890'
stop_words = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves",'a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always','among','an','and','another','any','anybody','anyone','anything','anywhere','are','area','areas','around','as','ask','asked','asking','asks','at','away','b','back','backed','backing','backs','be','became','because','become','becomes','been','before','began','behind','being','beings','best','better','between','big','both','but','by','c','came','can','cannot','case','cases','certain','certainly','clear','clearly','come','could','d','did','differ','different','differently','do','does','done','down','down','downed','downing','downs','during','e','each','early','either','end','ended','ending','ends','enough','even','evenly','ever','every','everybody','everyone','everything','everywhere','f','face','faces','fact','facts','far','felt','few','find','finds','first','for','four','from','full','fully','further','furthered','furthering','furthers','g','gave','general','generally','get','gets','give','given','gives','go','going','good','goods','got','great','greater','greatest','group','grouped','grouping','groups','h','had','has','have','having','he','her','here','herself','high','high','high','higher','highest','him','himself','his','how','however','i','if','important','in','interest','interested','interesting','interests','into','is','it','its','itself','j','just','k','keep','keeps','kind','knew','know','known','knows','l','large','largely','last','later','latest','least','less','let','lets','like','likely','long','longer','longest','m','made','make','making','man','many','may','me','member','members','men','might','more','most','mostly','mr','mrs','much','must','my','myself','n','necessary','need','needed','needing','needs','never','new','new','newer','newest','next','no','nobody','non','noone','not','nothing','now','nowhere','number','numbers','o','of','off','often','old','older','oldest','on','once','one','only','open','opened','opening','opens','or','order','ordered','ordering','orders','other','others','our','out','over','p','part','parted','parting','parts','per','perhaps','place','places','point','pointed','pointing','points','possible','present','presented','presenting','presents','problem','problems','put','puts','q','quite','r','rather','really','right','right','room','rooms','s','said','same','saw','say','says','second','seconds','see','seem','seemed','seeming','seems','sees','several','shall','she','should','show','showed','showing','shows','side','sides','since','small','smaller','smallest','so','some','somebody','someone','something','somewhere','state','states','still','still','such','sure','t','take','taken','than','that','the','their','them','then','there','therefore','these','they','thing','things','think','thinks','this','those','though','thought','thoughts','three','through','thus','to','today','together','too','took','toward','turn','turned','turning','turns','two','u','under','until','up','upon','us','use','used','uses','v','very','w','want','wanted','wanting','wants','was','way','ways','we','well','wells','went','were','what','when','where','whether','which','while','who','whole','whose','why','will','with','within','without','work','worked','working','works','would','x','y','year','years','yet','you','young','younger','youngest','your','yours','z'])

def get_file_name(id_to_file_name, id):
    return id_to_file_name[int(id)]

def get_words(utter):
    words = ''.join(i for i in utter if i not in stop_char)
    return words.split()

def get_main_words(utter):
    words = get_words(utter)
    words = [x for x in words if x not in stop_words and len(x) >= 3]
    return words

def getAcousticFeatures():
    ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
    ACOUSTIC_FEATURE = "../data/as1/features/AcousticFeatures/"
    id_to_file_name = {}
    results = []
    labels = []
    FEATURE_DIR = ACOUSTIC_FEATURE
    data = np.genfromtxt(ANNOTATED_SENTI,
                         delimiter=",", skip_header=1, usecols=[0, 1, 2, 3, 4, 5, 9])

    for fn in listdir(FEATURE_DIR):
        id = fn[5:7]
        if id[1] == "(":
            id = int(str(id[0]))
        id_to_file_name[int(str(id))] = fn
    
    # All acoustic
    feat_id_to_name = {4:"energy-dB",5:"energy-slope",3:"frequency",2:"NAQ",1:"PeakSlope",6:"stationarity",0:"Voiced-Unvoiced"}
    #feat_id_to_name = {2: "NAQ"}
    
    for i in feat_id_to_name.keys():
        # video feature file to be read
        video = ""
        result = []
        # Features in individual class
        class_negative_feature = []
        class_positive_feature = []
        class_neutral_feature = []
        for d_row in data:
            current_video = FEATURE_DIR + get_file_name(id_to_file_name, d_row[0])
            if (video != current_video):
                video = open(current_video, "r")
            start_ms = int(d_row[1] * 100) * 10
            end_ms = int(d_row[2] * 100) * 10
            label = int(d_row[3])
            pool_feature = []
            for obs_range in range(start_ms, end_ms, 10):
                features = video.readline().strip().split(",")
                feat_val = float(features[i])
    
                if str(feat_val)[-3:] == "inf" or str(feat_val) == "nan" or (abs(feat_val) > 18 and i == 1):
                    print feat_val, label
                    continue
                pool_feature.append(feat_val)
            if label == -1:
                class_negative_feature.append(np.std(pool_feature))
            elif label == 1:
                class_positive_feature.append(np.std(pool_feature))
            else:
                class_neutral_feature.append(np.std(pool_feature))
            result.append(np.std(pool_feature))
        results.append(result)
        labels.append("Feat-" + str(feat_id_to_name[i]))
    
    ANNOTATED_SENTI = "../data/as1/annotations/sentiment/sentimentAnnotations_rev_v03.csv"
    
    TRANSCRIPT_FEATURE = "../data/as1/annotations/Transcriptions/"
    
    id_to_file_name = {}
    
    FEATURE_DIR = TRANSCRIPT_FEATURE
    
    data = np.genfromtxt(ANNOTATED_SENTI,
                         delimiter=",", skip_header=1, usecols=[0,1,2,9])
    
    for fn in listdir(FEATURE_DIR):
        vid_id = fn[5:7]
        if vid_id[1]=="(":
            vid_id=int(str(vid_id[0]))
        id_to_file_name[int(str(vid_id))]=fn
    
    video_to_transcript={}
    
    last_uttsindex = 1
    
    result_elong = []
    result_obs = []
    for d_row in data:
        vid_new = FEATURE_DIR + get_file_name(id_to_file_name,d_row[0])
        end_ms = int(d_row[2]*1000)

        if vid_new in video_to_transcript:
            events,utts = video_to_transcript[vid_new]
        else:
            print vid_new
            last_uttsindex = 1
            events,utts = readTrans(vid_new)
            video_to_transcript[vid_new] = events,utts
    
        #         print label, last_uttsindex, end_ms
        feat_val_elong = []
        feat_val_obs = []
        while last_uttsindex in events:
            if events[last_uttsindex][1]*1000 > end_ms:
                break
            # total elongation in a utterance
            feat_val_elong.append( sum([ 1 for char in utts[last_uttsindex] if char == ':']) )
    
            # Obscenity count
            feat_val_obs.append( profanity_count(get_words(utts[last_uttsindex])) )
    
    
            last_uttsindex+=1
    
        if feat_val_elong == []:
            feat_val_elong = [0]
        if feat_val_obs == []:
            feat_val_obs = [0]
        feat_val_elong = [np.std(feat_val_elong)]
        feat_val_obs = [np.std(feat_val_obs)]
    
        result_elong += feat_val_elong
        result_obs += feat_val_obs
    results.append(result_elong)
    results.append(result_obs)
    labels.append("TRS-Elongation")
    labels.append("TRS-Obscene count")
    return results,labels