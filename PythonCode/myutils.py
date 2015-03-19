import numpy as np

def read_tins_files(ut_set='tr'):
    import os
    mfcc_files = []
    
    spk_set = ['mdab0','mwbt0','felc0',
               'mtas1','mwew0','fpas0',
               'mjmp0','mlnt0','fpkt0',
               'mlll0','mtls0','fjlm0',
               'mbpm0','mklt0','fnlp0',
               'mcmj0','mjdh0','fmgd0',
               'mgrt0','mnjm0','fdhc0',
               'mjln0','mpam0','fmld0']

    if ut_set == 'tr':
        root_path=os.environ['TIMIT']+'/train'    
        for dr in os.listdir(root_path):
            for spk in os.listdir(os.path.join(root_path,dr)):
                limit = 1
                for f in os.listdir(os.path.join(root_path,dr,spk)):
                    if ( f.endswith('.mfcc') and not f.startswith('sa') and limit):
                        mfcc_files.append(os.path.join(root_path,dr,spk,f))
                        limit = 0
    #Test
                        
    elif ut_set=='tst':
        root_path=os.environ['TIMIT']+'/test'
        for dr in os.listdir(root_path):
            for spk in os.listdir(os.path.join(root_path,dr)):
                if spk in spk_set:
                    limit = 1
                    for f in os.listdir(os.path.join(root_path,dr,spk)):
                        if (f.endswith('.mfcc') and not f.startswith('sa') and limit):
                            mfcc_files.append(os.path.join(root_path,dr,spk,f))
                            limit = 0
    else:
        limit = 0
        root_path=os.environ['TIMIT']+'/test'
        for dr in os.listdir(root_path):
            for spk in os.listdir(os.path.join(root_path,dr)):
                if (spk not in spk_set  and limit < 50):
                    limit_f = 1 
                    for f in os.listdir(os.path.join(root_path,dr,spk)):
                        if (f.endswith('.mfcc') and not f.startswith('sa') and limit_f):
                            mfcc_files.append(os.path.join(root_path,dr,spk,f))
                            limit_f = 0
                    limit+=1

                                                      
    return mfcc_files



def read_tins_data(tins_dir_path,train=True, p=0.2):
    n_phonemes = 61
    frame_cols = tuple([i for i in xrange(13)])
  
    if train:
        data = np.loadtxt(tins_dir_path+'train.csv',dtype=np.float32,delimiter=',',usecols=frame_cols)
        labels = np.loadtxt(tins_dir_path+'trainL.csv',dtype=np.int32,delimiter=',')
        n = data.shape[0]
        if p>0:
            perm = np.random.permutation(n)

        mean_tr = data.mean(axis=0)
        std_tr  = data.std(axis=0)
        data_norm = (data-mean_tr)/std_tr
    
        #if p>0:
        #    tr_end_idx = np.floor(n*(1-p))
        #else:
        #    tr_end_idx = data.shape[0]+1
    
        #perm_tr  = perm[0:tr_end_idx]
        #perm_dev = perm[tr_end_idx:]
    
        stacked_features_tr  = data_norm #[perm_tr,]
        #stacked_features_dev = data_norm[perm_dev,] 
    
        stacked_labels_tr = labels#[perm_tr,]
       # stacked_labels_dev = labels[perm_dev,]
    
        phoneme_priors = np.zeros(n_phonemes,dtype='float32')
        for j in labels:
            phoneme_priors[j] += 1

            prior_tr = phoneme_priors/np.sum(phoneme_priors)
    
        return  stacked_features_tr,stacked_labels_tr,prior_tr,mean_tr,std_tr
    else:
        data = np.loadtxt(tins_dir_path+'test.csv',dtype=np.float32,delimiter=',',usecols=frame_cols)
        labels = np.loadtxt(tins_dir_path+'testL.csv',dtype=np.int32,delimiter=',')
        return  data,labels

def build_phn_dict():
    dictionary = []
    #Folding and mapping to 39 phonemes
    for i in xrange(61):
        
        if i==45:
            val=29
        elif i==60:
            val=46
        elif i==53:
            val=23
        elif i==54:
            val=27
        elif i==57:
            val=13
        elif i==58:
            val=0
        elif i==42 or i==16:
            val=5
        elif i==38 or i==33:
            val=10
        elif i==40:
            val=3
        else:
            val=i
        
        dictionary.append(val)
    return dictionary

#Encoding the levenshtein string to chars
def build_lev_dict():
    dictionary = {}
    #silence is mapped to space 
    dictionary[0]=' '
    for i in xrange(60):
        dictionary[i+1]=chr(48+i)
    
    return dictionary

phn_map = build_phn_dict()
lev_string_map_dict = build_lev_dict()

def compute_per(sols, stacked_labels_tst, frame_idces_tst):
    from Levenshtein import editops,distance
    err = []
    diff = 0
    errs = []
    for i in xrange(len(sols)): 
        hmm_pred = []
        y_seq = []
        for j in xrange(len(sols[i])):
            hmm_pred.append(  lev_string_map_dict[ phn_map[sols[i][j]] ] )
            y_seq.append(lev_string_map_dict [phn_map[stacked_labels_tst[frame_idces_tst[i][0]:frame_idces_tst[i][1]+1][j]] ])
            
   #     print hmm_pred
   #     print y_seq
   #     print "============="
            
        hmm_pred_str=''.join(hmm_pred).strip()
        y_seq_str=''.join(y_seq).strip()
        
        ops = editops(hmm_pred_str,y_seq_str)
        n_ops = [0 if i[0]=='insert' else 1 for i in ops ]
        errs.append(float(sum(n_ops))/len(y_seq_str))
            
    return 100*np.mean(errs)


def process_forward_data(f_res,test_set_x,test_set_y,
                         batch_size, n_phonemes, n_layers):
        
    n_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size
    frames = test_set_x.get_value()
    labels = test_set_y.get_value()
    data = [[] for i in xrange(n_phonemes)]
    batch_errors = []

    for batch_idx in xrange(n_batches):
        batch = f_res[batch_idx]

        for frame_idx in xrange(batch_size):
            #frame_info has  [frame, rbm layers, sigm_layers]
            #frame_info = []
            #frame_info.append(frames[batch_idx*batch_size+frame_idx,])
            #frame_info.append(labels[batch_idx*batch_size+frame_idx,])
            data_idx=labels[batch_idx*batch_size+frame_idx]
            #rbm layers
            rbm_layers = []
            sigm_layers = []
            for i in xrange(n_layers):
                rbm_layers.append(batch[i][frame_idx])
                #sigm_layers.append(batch[n_layers+i][frame_idx])
            
            frame_info=rbm_layers#.append(rbm_layers)
            #frame_info.append(sigm_layers)
            #classified label
            #frame_info.append(batch[-1][frame_idx])
            data[data_idx].append(frame_info)
        batch_errors.append(batch[-1])
    #return features info and batch errors
    return data,batch_errors


def read_mfcc_files(root_path, spk_set=None):
    import os
    mfcc_files = []
    if (spk_set == None):
        for dirpath,dirnames,filenames in os.walk(root_path):
            mfcc_files.extend([os.path.join(dirpath,f) for f in filenames if (f.endswith('.mfcc') and not f.startswith('sa'))])
    else:
        for dr in os.listdir(root_path):
            for spk in os.listdir(os.path.join(root_path,dr)):
                if spk in spk_set :
                    mfcc_files.extend([os.path.join(root_path,dr,spk,f) 
                                       for f in os.listdir(os.path.join(root_path,dr,spk)) 
                                       if (f.endswith('.mfcc') and not f.startswith('sa'))])
                
    return mfcc_files

def read_dev_files(root_path, excl_set,n_spk):
    import os
    mfcc_files = []
    limit = 0
    for dr in os.listdir(root_path):
        for spk in os.listdir(os.path.join(root_path,dr)):
            if (spk not in excl_set and limit<n_spk):
                mfcc_files.extend([os.path.join(root_path,dr,spk,f) 
                                   for f in os.listdir(os.path.join(root_path,dr,spk)) 
                                   if (f.endswith('.mfcc') and not f.startswith('sa'))])
                limit+=1
    return mfcc_files




#Reads information directly from .mfcc
def process_mfcc_files(mfcc_files,n_phonemes,n=None,mean=None,std=None,n_states_per_phoneme=3):
    frame_cols=tuple([i for i in xrange(13)])
    n_utterances = n if (n != None) else len(mfcc_files)
    frames_ar = []
    labels_ar = []
    idx_init=0
    phoneme_priors = np.zeros(n_phonemes,dtype='float32')
    frame_idces = []

    for i in xrange(n_utterances):
        #print mfcc_files[i]
        frame_i = np.loadtxt(mfcc_files[i],dtype=np.float32,delimiter=' ',usecols=frame_cols)
        frames_ar.append(frame_i)
        
        frame_idces.append( (idx_init , idx_init + frame_i.shape[0]-1) )
        idx_init += frame_i.shape[0]

        label_i = np.loadtxt(mfcc_files[i],dtype=np.int32,delimiter=' ',usecols=[13])
        labels_ar.append(label_i)
        
        #sum to calc. priors
        for j in label_i:
            phoneme_priors[j] += 1
            
    
    #Stack features as a matrix
    stacked_frames = np.vstack(frames_ar)
    
    #Stack labels
    stacked_labels = np.concatenate(labels_ar)

    #Stack idces
    stacked_idces = np.vstack(frame_idces)
    
   
    
    #Normalize frames only if training
    if (mean == None and std == None):
        mean = np.mean(stacked_frames,axis=0)
        std  = np.std(stacked_frames,axis=0)
        
    
    stacked_frames = (stacked_frames - mean)/std
    
        
        
        
    phoneme_priors = phoneme_priors/np.sum(phoneme_priors)
    for i in xrange(n_phonemes):
        if (phoneme_priors[i] == 0):
            phoneme_priors[i] = 1e-4
    
    phoneme_priors = np.reshape(np.tile(phoneme_priors,(n_states_per_phoneme,1)),(n_phonemes*n_states_per_phoneme,),'F')
       
    return stacked_frames, stacked_labels, stacked_idces, phoneme_priors,mean,std
    
#Gets files path of the data base
def read_files(root_path):
    import os
    audio = []
    labels = []
    
    replace_wav_lbl = lambda x : x.replace('.wav','.lbl')
    for dirpath,dirnames,filenames in os.walk(root_path):
        audio = audio + [os.path.join(dirpath,f) for f in filenames if (f.endswith('.wav') and not f.startswith('sa'))]
        labels = map(replace_wav_lbl,audio)
    audio.sort()
    labels.sort()
    return audio,labels     





#Reads n audio files with phonemes labels
def read_data(audio_files,label_files,n_phonemes=61,n=None): 
    #import theano
    import scikits.audiolab as audiolab
    from scikits.talkbox.features import mfcc
   # import mfcc 
    
    n_audios = n if (not n) else len(audio_files)
    cepstra_ar = []
    labels_ar = []
    phoneme_priors = np.zeros(n_phonemes,dtype='float32')
    frame_end_idces = []
    idx_acc=0

    for i in np.random.permutation(n_audios)[:n]:
        f = audiolab.sndfile(audio_files[i],'read')
        audio_signal = f.read_frames(f.get_nframes())
        cep = mfcc(audio_signal)[0]
        #normalize frames
        #cep = (cep - np.mean(cep,axis=0))/np.std(cep,axis=0)
        cep_nframes = cep.shape[0]
        idx_acc += cep_nframes
        frame_end_idces.append(np.array([idx_acc-cep_nframes,idx_acc-1], dtype='int32'))
        cepstra_ar.append(np.asarray(cep,dtype='float32'))
        f.close()
        
        #load labels
        labels_info = np.loadtxt(label_files[i],dtype=np.int32,delimiter=' ',usecols=(0, 1, 2))
        labels = process_labels_info(cep_nframes, labels_info)
        labels_ar.append(labels)
        
        #sum to calc. priors
        for j in labels:
            phoneme_priors[j] += 1



    #Stack features as a matrix
    stacked_frames = np.vstack(cepstra_ar)
    
    #Stack idces
    stacked_idces = np.vstack(frame_end_idces)

    #Normaliza frames
    stacked_frames = (stacked_frames - np.mean(stacked_frames,axis=0))/np.std(stacked_frames,axis=0)

    #Stack labels
    stacked_labels = [] 
    [stacked_labels.extend(i) for i in labels_ar]
    stacked_labels = np.asarray(stacked_labels)

    

    phoneme_priors=phoneme_priors/np.sum(phoneme_priors)
    
    #Avoid zeros
    for i in xrange(n_phonemes):
        if (phoneme_priors[i] == 0):
            phoneme_priors[i] = 1e-10

    return stacked_frames, stacked_labels, stacked_idces, phoneme_priors




def process_labels_info(n_frames, labels_info):
    displacement=labels_info[0][0]
    labels=[labels_info[ np.nonzero(displacement + i*160 >= labels_info[:,0])[0][-1], 2] for i in xrange(n_frames)]
    return labels

#fixed for a max of 5 GB of GPU Mem
def get_batch_size(n, max_batch=120000):
    lim = np.amin([n, max_batch])
    for i in xrange(lim, 1, -1):
        if (n%i == 0):
            return i
        
    return 1

#Extends the DNN input fron 1 frame to n frames adding context.
def extend_input(stacked_frames, stacked_idces, n):
    assert n%2 == 1 , "Number of extension must be odd!"
    context = n/2
    extended_frames = np.zeros([stacked_frames.shape[0], 
                                stacked_frames.shape[1]*n],dtype = 'float32')
    ext_idx = 0
    for ut_idx in xrange(stacked_idces.shape[0]):
        idx_tuple = stacked_idces[ut_idx]
        ut_frames = stacked_frames[idx_tuple[0]:idx_tuple[1]+1,]
        
        #extend the initial and last frames
        ext_frame = np.vstack( [np.tile(ut_frames[0,],(context,1)) , ut_frames])
        ext_frame = np.vstack( [ext_frame, np.tile(ut_frames[-1,],(context,1))])
        
        #extend the cols
        for j in xrange(ut_frames.shape[0]):
            window = ext_frame[j:j+n,]

            extended_frames[ext_idx+j,] = np.reshape(window,(1, n*ut_frames.shape[1]))
        ext_idx += ut_frames.shape[0]
        
    
    return extended_frames

def build_initial_hmm_mat(n_states_per_phoneme, n_phonemes):
    n_states = n_phonemes * n_states_per_phoneme
    pi = np.zeros(n_phonemes,dtype='float32')#np.asarray([1./(n_phonemes) if (i% n_states_per_phoneme == 0) else 0 for i in xrange(n_states)], dtype='float32')
    pi[0]+=1.
    A = [] 
    
    if (n_states_per_phoneme==1):
        row = [1./(n_phonemes) if (i% n_states_per_phoneme == 0) else 0 for i in xrange(n_states)]
        for i in xrange(n_states):
            A.append(row)
    else:
    
        for i in xrange(n_states):
            if (i%n_states_per_phoneme == 0 or i%n_states_per_phoneme == 1):
                A.append([0.5 if (j==i or j==i+1 ) else 0 for j in xrange(n_states)]) 
            
            if (i%n_states_per_phoneme == 2):
                A.append([1./(n_phonemes+1) if (j==i or j%n_states_per_phoneme == 0) else 0 for j in xrange(n_states)])
            
    A = np.asarray(A, dtype='float32')
    return pi,A



import os
#print len(read_tins_files('tr'))

#tins_dir_path= os.environ['TIMIT']+'/tin/'
#stacked_features_tr,stacked_labels_tr,stacked_features_dev,stacked_labels_dev,prior_tr,mean_tr,std_tr = read_tins_data(tins_dir_path,train=True, p=0.2)
#print stacked_features_tr.shape
#print stacked_labels_tr.shape
#print stacked_features_dev.shape
#print stacked_labels_dev.shape
#pi,A = build_initial_hmm_mat(3, 3)
#print pi
#print A

#
##import os
#print read_dev_files(os.environ['TIMIT']+'/test',core_test_set,n_spk=50)

#print len(read_mfcc_files('/home/nizar/timit/timit/timit/test',spk_set=core_test_set))



#x = np.array([[1,2,3,4,5],[2,2,2,4,5],[7,2,8,4,5],[3,3,3,3,3],[1,1,1,1,1],
#              [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],
#              [7,7,7,7,7],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],
#              [1,2,3,4,5],[1,2,3,4,5],[8,8,8,8,8],[1,9,8,5,5],[1,2,3,4,5]])
#idces = np.array([[0,4],[5,9],[10,14],[15,19]])
#print x
#print extend_input(x,idces, 5)


#mfcc_files = read_mfcc_files('/home/nizar/timit/timit/timit/train')
#stacked_frames, stacked_labels, stacked_idces, phoneme_priors = process_mfcc_files(mfcc_files,61,n=10)
#print mfcc_files[0]
#print stacked_frames
#print stacked_labels
#print stacked_idces
#print phoneme_priors


#print get_batch_size(3648)
#import time as t
#t3=t.time()
#audio_f,labels_f=read_files('/home/nizar/timit/timit/timit/train')
#signals,labels,phoneme_priors=read_data(audio_f,labels_f,n=500)
#t4=t.time()
#print t4-t3
#print sum(phoneme_priors)
#print phoneme_priors
#print signals[0].shape
#print len(labels[0])
#print labels[0]
#print labels[0]
#print signals[0].shape
#print signals[0][0].shape
#print labels[0]
