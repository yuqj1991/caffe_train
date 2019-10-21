import logging
import pandas as pd

#ori
ORI_BATCH1 = 'umdfaces_batch1/'
ORI_BATCH2 = 'umdfaces_batch2/'
ORI_BATCH3 = 'umdfaces_batch3/'

#process
PROCESSED_BATCH1 = '../../../dataset/facedata/umdface/images/umdfaces_batch1/'
PROCESSED_BATCH2 = '../../../dataset/facedata/umdface/images/umdfaces_batch2/'
PROCESSED_BATCH3 = '../../../dataset/facedata/umdface/images/umdfaces_batch3/'

def init_my_logger(): 
    logger = logging.getLogger()  
    logger.setLevel(logging.DEBUG)
    
    logfile = './log.txt'  
    fh = logging.FileHandler(logfile, mode='w')  
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")  
    fh.setFormatter(formatter)  
    ch.setFormatter(formatter)  
    
    logger.addHandler(fh)  
    logger.addHandler(ch)  

    return logger

def read_from_file(file_name, chunk_size=500000):
    reader = pd.read_csv(file_name, iterator=True)
    chunks = []
    mark = True
    while mark:
        try:
            df = reader.get_chunk(chunk_size)
            chunks.append(df)
        except:
            print "Iterator Stop..."
            mark = False
    df = pd.concat(chunks,ignore_index=True)
    return df
