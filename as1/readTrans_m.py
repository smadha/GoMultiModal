
def readTrans(file_name):
    events=[]
    utts={}
    
    with open(file_name,"r") as trans_file:
        for i in range(0,7):
            trans_file.readline().strip()
        line = trans_file.readline().strip()
        line_1 = trans_file.readline().strip()
        line_2 = trans_file.readline().strip()
        uttsindex=1 #                                            %index of sentences
    
        while line != "":
            if ( (line.startswith('<'))  #!isempty(strmatch('<',seq{line})))
                        and (not line_1.startswith('<')) #and (isempty(strmatch('<',seq{line+1})) 
                        and (not line_1 =='((silence))' ) #and !(strcmp(seq{line+1},'((silence))')) 
                        and (not line_1 =='((long pause))' )  #and !(strcmp(seq{line+1},'((long pause))')) 
                        and (not line_1 =='(intro)' )  #and !(strcmp(seq{line+1},'(intro)')) 
                        and (not line_2 =='</Turn>' ) ):  #and !(strcmp(seq{line+2},'</Turn>')))
                
                # sample line - <Sync time="2.94"/>
                begin = line[12:line.index("/")-1] 
                
                stop = line[12:line.index("/")-1]
                
    #             events{i}(uttsindex,1)=str2double(begin)-OffsetDing(i);
    #             events{i}(uttsindex,2)=str2double(stop)-OffsetDing(i);
    #             events{i}(uttsindex,3)=uttsindex;
                events.append([float(begin),float(stop),uttsindex])
    
    #             utterance(numel(seq{line+1})+1:300)=' ';        %utterance is sentence(max 300 char)
    #             utterance(1:numel(seq{line+1}))=seq{line+1};
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