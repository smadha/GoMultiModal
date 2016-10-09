

def readTrans(file_name):
    '''
    Open one TRS file. Converted from readTrans.m
    '''
    events={}
    utts={}
    
    with open(file_name,"r") as trans_file:
        for i in range(0,6):
            trans_file.readline().strip()
        line = trans_file.readline().strip()
        line_1 = trans_file.readline().strip()
        line_2 = trans_file.readline().strip()
        uttsindex=1 #                                            %index of sentences
        
        while line != "" or line_1 != ""  or line_2 != "" :
            if ( (line.startswith('<'))  
                        and (not line_1.startswith('<'))  
                        and (not line_1 =='((silence))' ) 
                        and (not line_1 =='((long pause))' )   
                        and (not line_1 =='(intro)' )   
                        and (not line_2 =='</Turn>' ) ): 
                
                # sample line - <Sync time="2.94"/>
                begin = line[12:line.index("/")-1] 
                
                stop = line_2[12:line_2.index("/")-1]
                
                events[uttsindex] = [float(begin),float(stop)]
    
                utterance=[' ']*300
                utterance[0:len(line_1)]=line_1
                
                utts[uttsindex]=utterance
                uttsindex=uttsindex+1
                    
            
            line = line_1
            line_1 = line_2
            line_2 = trans_file.readline().strip()
            
            if (line == "</Turn>"):
                break
            
    return events,utts

if __name__ == '__main__':
    print readTrans("../data/as1/annotations/Transcriptions/video10(00h00m29s-00h00m58s).trs")