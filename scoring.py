import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from model import vggvox_model,vggvox_model_output512
from wav_reader import get_fft_spectrum
import constants as c
import pickle


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)          #FRAME_STEP = 0.01
    end_frame = int(max_sec*frames_per_sec)     #MAX_SEC = 10
    step_frame = int(step_sec*frames_per_sec)   #BUCKET_STEP = 1
    for i in range(0, end_frame+1, step_frame):
        s = i  #0,100,200,300,400,...,900
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def get_embedding(model, wav_file, max_sec):
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    signal = get_fft_spectrum(wav_file, buckets)
    embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
    return embedding


def get_embedding_batch(model, wavpath, max_sec):
        return get_embedding(model, wavpath, max_sec)

'''
def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]
'''

def get_id_result():
    print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
    #model = vggvox_model()
    model = vggvox_model_output512()
    model.load_weights(c.WEIGHTS_FILE)
    print('finish loding model----------------------------')
    #model.summary()
    
    '''
    print("Processing enroll samples....")
    enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
    enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
    speakers = enroll_result['speaker']
    '''
    record={}
    n=0
    print("Processing test samples....")
    #test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
    #test_embs = np.array([emb.tolist() for emb in test_result['embedding']])
    for wav_file in os.listdir(wav_files):
        wavpath = os.path.join(wav_files,wav_file)
        #print('wave name:',wav_file)
        video_name = wav_file.split('.')[0]
        #print('path:',wavpath)
        try:
            test_result = get_embedding_batch(model,wavpath,c.MAX_SEC)
        except:
            test_result = np.zeros((512,))
        #print('test_result:',test_result.shape) #(1024,)
        #print('video_name:',video_name) #IQIYI_VID_TRAIN_0262102
        n+=1
        record[str(video_name)]=test_result
        if n%100==0:
            print('n:',n)

    #print('record:',record)
    with open('ye512_audio_val.pickle','wb') as f:
        pickle.dump(record,f)


    '''
    print("Comparing test samples against enroll samples....")
    distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)

    scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
    scores = pd.concat([scores, distances],axis=1)
    scores['result'] = scores[speakers].idxmin(axis=1)
    scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

    print("Writing outputs to [{}]....".format(c.RESULT_FILE))
    result_dir = os.path.dirname(c.RESULT_FILE)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(c.RESULT_FILE, 'w') as f:
        scores.to_csv(f, index=False)
    '''


if __name__ == '__main__':
    wav_files='/home/data_warehouse/ansix/aiqiyi/2019_dataset/IQIYI_VID_DATA_2019/IQIYI_2019_WAV_VAL/'
    #wav_files='/home/data_warehouse/ansix/aiqiyi/2019_dataset/IQIYI_VID_DATA_2019/IQIYI_2019_WAV_TRAIN/'
    get_id_result()
