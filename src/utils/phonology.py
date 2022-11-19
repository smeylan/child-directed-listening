import unicodedata

def phone_remap(x):
    # this is the one defined inline in 1 - Create Vocabuary; probably the best one

    return(x.replace("ː","").replace('ʌ','ə').replace('ɪ','ə').replace('ɔ','ɑ').replace('˞','')\
           .replace('ʰ','').replace('r','ɹ').replace('^','')\
          .replace('ʙ','b').replace('c','k').replace('g','ɡ')\
          .replace('y','j').replace('ʁ','ɹ').replace('(','')\
          .replace(')','').replace('.',''))


# def phone_remap(x):    

# 	# this is the one that was in load_models.py
#     # this is actually not called any more because get_cmu_dict is processed elsewhere

#     return(x.replace("ː","").replace('ʌ','ə')
# .replace('ɪ','ə').replace('ɔ','ɑ').replace('a','ɑ').replace('o','oʊ').replace('˞','').replace('ʰ',
#     ''). replace('r','ɹ')).replace('\\^','').replace('\\ ̃','').replace(' ̩','').replace('^',''
# ).replace('ʙ','b').replace('(','').replace(')','').replace('.','').replace('ch','ʧ'
# ).replace('c','k').replace('g','ɡ').replace('y','j').replace('ʁ','ɹ')

# def phone_remap(x):
#     # From Providence Retrieve data
#     return(x.replace("ː","").replace('ʌ','ə').replace('ɪ','ə').replace('ɔ','ɑ').replace('˞','')\
#            .replace('ʰ','').replace('r','ɹ').replace('^','')\
#           .replace('ʙ','b').replace('c','k').replace('g','ɡ')\
#           .replace('y','j').replace('ʁ','ɹ').replace('(','')\
#           .replace(')','').replace('.',''))    

def strip_accents(string, accents=('COMBINING ACUTE ACCENT', 
    'COMBINING GRAVE ACCENT', 'COMBINING TILDE', 'COMBINING VERTICAL LINE BELOW',
    'COMBINING SHORT STROKE OVERLAY')):
    accents = set(map(unicodedata.lookup, accents))
    chars = [c for c in unicodedata.normalize('NFD', string) if c not in accents]
    return unicodedata.normalize('NFC', ''.join(chars))
