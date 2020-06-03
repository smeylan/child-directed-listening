import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from io import StringIO
import re
import os

def check_art(word,method):    
    if list(word)[-1] in ('s','z','ð','θ','ʒ'):
        return(True)
    if method == 'model':
        # already did the one appropriate check, above
        return(False)
    elif method == 'actual':        
        secondSylsz = re.findall("^.*ˈ.*[szðθʒ].+", word)
        # alternatively, find the number of s
        if len(secondSylsz) > 0:
            return(True)
        else:
            return(False)

def robust_nan_check(x):
    try:
        if np.isnan(x): 
            return None
        else: 
            return(x)            
    except:
        return(x)

def extractFromUtterance(utterance, verbose=False):
    transcribed_words = [x.text for x in utterance.findall(".//ipaTier[@form='actual']/pg/w") if x.text is not None]
    #morph_words = [x.text for x in utterance.findall(".//groupTier[@tierName='Morphology']/tg/w") \
    #    if not x.text in (None,'.')]
    model_words = [x.text for x in utterance.findall(".//ipaTier[@form='model']/pg/w") if x.text is not None]

    model_words = [robust_nan_check(x) for x in model_words if x != '(.)' and x is not None]
    transcribed_words = [robust_nan_check(x) for x in transcribed_words if x != '(.)' and x is not None]
    
    # get the glosses
    gloss_list = [x.text for x in utterance.findall(".//orthography/g/w")] 
    gloss_list = [x for x in gloss_list if not '0' in x]# drop 0-marked words from the gloss for alignment
    gloss = ' '.join(gloss_list)
    
    if (len(transcribed_words)) > 0 & (len(gloss_list) > 0):
        if (len(transcribed_words) != len(model_words)) & verbose:        
            print('length mismatch!')
            print(transcribed_words)
            print(model_words)
            length_mismatch = True
        else:
            length_mismatch = False
        
        while len(model_words) < len(transcribed_words):
            model_words.append(np.nan)
            
        while len(transcribed_words) < len(model_words):
            transcribed_words.append(np.nan) 

        
        if len(gloss_list) != len(model_words):            
            return(None) # dump it if the gloss doesn't line up with the model words


        # try    
        rdf = pd.DataFrame({'actual':transcribed_words, 
        'model':model_words, 'word_gloss': gloss_list})        
        rdf['preceding_gloss'] = [gloss_list[0:i] for i in range(rdf.shape[0])]
        rdf['spk'] = utterance.attrib['speaker']
        rdf['gloss'] = gloss
        rdf['transcribed_model_length_mismatch'] = length_mismatch
        return(rdf)
        # except:
        #     print('problem with extraction!')
        #     import pdb
        #     pdb.set_trace()                    
        

def getAgeFromDatestr(datestr, mode):    
    if mode == 'id':
        # doesn't always work
        if len(datestr) != 6:
            idString = idString[0:6]
            #raise ValueError('id string does not have 6 digits')

        years = int(idString[0:2])
        months = int(idString[2:4])
        days = int(idString[4:6])
        
    elif mode == 'bmw':
        years = int(re.findall("P(.+?)Y", datestr)[0])        
        months = int(re.findall("Y(.+?)M", datestr)[0])
        days = int(re.findall("M(.+?)D", datestr)[0])

    if years > 20:
        print(idString)
        raise ValueError('Years out of range')        

    
    if months > 12:
        print(idString)
        raise ValueError('months out of range')        
    
    
    if days > 31:
        print(idString)
        raise ValueError('Days out of range')
    return((years*30.5*12) + round(30.5*months) + days)        
    

def getArticulationProp(xml_path):
    print('Processing '+xml_path+'...')
    with open(xml_path, 'r') as file:
        xml = file.read()

    # must get rid of the namespace before doing anything else
    it = ET.iterparse(StringIO(xml)) 
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
    root = it.root

    utts = [extractFromUtterance(utterance) for utterance in root.findall("*/u")]

    numbered_utts = []
    for i in range(len(utts)):
        df = utts[i]
        if df is not None:
            if df.shape[0] > 0:
                df['utt_index'] = i
                numbered_utts.append(df)
    if len(numbered_utts) > 0:
        tr_df = pd.concat(numbered_utts).dropna()
        
        tr_df['citation_form_requires'] =  [check_art(x,'model') for x in tr_df['model']]
        # this is where I may be able to recover the index

        citation_form_requires = tr_df.loc[tr_df.citation_form_requires]
        citation_form_requires['actual_form_sz'] = [check_art(x,'actual') for x in citation_form_requires.actual]

        citation_form_requires['age'] = getAgeFromDatestr(root.find(".//*[@id='CHI']/age").text ,'bmw')
        citation_form_requires['child'] = root.attrib['corpus']
        citation_form_requires['xml_path'] = os.path.basename(xml_path)
        return(citation_form_requires)
    else:
        return(None)
