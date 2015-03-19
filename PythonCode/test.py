#!/usr/bin/env python

import time as t
import myutils
import DeepLearningTB.DBN as db
import numpy as np
import theano
import theano.tensor as T
from hmm import HMM,EmissionModel
import cPickle
import argparse
import os

def main():
    
    
    parser = argparse.ArgumentParser( description='Parameters Specification.')
    parser.add_argument("-v", help="Show output verbosity.", type=int)
    parser.add_argument("-e", help="Pretraining epochs.", type=int)
    #parser.add_argument("-tst", help="Number of test utterances to take.", type=int)
    parser.add_argument("-ntr", help="Forward the training.", type=int)
    parser.add_argument("-training", help="Forward the training.", type=int)
    parser.add_argument("-l", help="Number of hidden layers.", type=str)
    parser.add_argument("-m", help="Option to specify a model.", type=str)
    parser.add_argument("-per", help="Perform PER calculation.",type=int)
    parser.add_argument("-save", help="Save results information.",type=int)
    parser.add_argument("-tin", help="Use tin's dataset.", type=int)
    parser.add_argument("-rl", help="Use real layers.", type=int)
    
    args = parser.parse_args()
    
    utterances_tst = args.ntr if args.ntr is not None else None
    
    if args.l is not None :
        hidden_layers_sz = [int(i) for i in args.l.split(',')]
    else:
        hidden_layers_sz = [2048, 2048, 2048, 2048]

    dbn_model = args.m if args.m is not None else 'DBN_RBM_final.save'
    pretraining_epochs = args.e if args.e is not None else 225
    
    n_phonemes = 61
    n_states_per_phoneme = 1
    
    if args.tin:
        extend_input_n=1
    else:
        extend_input_n = 11
    
    
    models_base_path = os.environ['MT_ROOT']+'/Models/'
    if args.tin:
        model_dir_path = models_base_path+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+'_pr_'+str(pretraining_epochs)+'_tin/'
    else:
        model_dir_path = models_base_path+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+'_pr_'+str(pretraining_epochs)+'/'
    
    print 'Model: '
    print model_dir_path

    if args.v == 1:
         print '... loading pickled mean and std.'
    
    
    f = file(model_dir_path+'mean_std_pr.save', 'rb')
    mean_tr,std_tr,prior_tr = cPickle.load(f)
    f.close()
    
    core_test_set = ['mdab0','mwbt0','felc0',
                     'mtas1','mwew0','fpas0',
                     'mjmp0','mlnt0','fpkt0',
                     'mlll0','mtls0','fjlm0',
                     'mbpm0','mklt0','fnlp0',
                     'mcmj0','mjdh0','fmgd0',
                     'mgrt0','mnjm0','fdhc0',
                     'mjln0','mpam0','fmld0']
    
    if (args.training) : 
        if args.v == 1:
            print '... reading training files.'
        if args.tin:
            stacked_features_tst,stacked_labels_tst,prior_tst,mean_tst,std_tst = myutils.read_tins_data(os.environ['TIMIT']+'/tin/',p = 0)
            #mfcc_f_tst = myutils.read_tins_files('tr')
            #stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes, n=utterances_tst,mean=mean_tr,std=std_tr)
      
        else:
            mfcc_f_tst = myutils.read_mfcc_files(os.environ['TIMIT']+'/train')
            stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes, n=utterances_tst,mean=mean_tr,std=std_tr)
         #to make it dividable
            if args.ntr is None and args.tin is None:
                stacked_features_tst = stacked_features_tst[0:-3,]
                stacked_labels_tst   = stacked_labels_tst[0:-3,]
                frame_idces_tst[-1,1] -= 3

    
    else:
        if args.v == 1:
            print '... reading test files.'
        if args.tin : 
             #mfcc_f_tst = myutils.read_tins_files('tst')
             #stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes, n=utterances_tst,mean=mean_tr,std=std_tr)

            stacked_features_tst,stacked_labels_tst = myutils.read_tins_data(os.environ['TIMIT']+'/tin/',train=False)
        else:
            mfcc_f_tst = myutils.read_mfcc_files(os.environ['TIMIT']+'/test',spk_set=core_test_set)
            stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes, n=utterances_tst,mean=mean_tr,std=std_tr)

    
    if args.tin is None:
        stacked_features_tst = myutils.extend_input(stacked_features_tst, frame_idces_tst, extend_input_n)
    
    batch_size_tst     = myutils.get_batch_size(stacked_features_tst.shape[0])
    n_test_batches     = stacked_features_tst.shape[0] / batch_size_tst

    
    test_set_x = theano.shared(np.asarray(stacked_features_tst,dtype=theano.config.floatX),
                               name='test_set',borrow=True)
    
    test_set_y = theano.shared(np.asarray(stacked_labels_tst,dtype='int32'),
                               name='test_labels',borrow=True)
    
    if args.tin is None:
        utterances_idces_tst = theano.shared(np.asarray(frame_idces_tst, dtype='int32'),
                                             name='test_idces', borrow=True)
    
    if args.v >= 1 :
        print '====Test===='
        if args.tin is None:
            print 'Total utterances: ' + str(frame_idces_tst.shape[0])
        print 'Total MFCCs: ' + str(stacked_features_tst.shape[0])
        print 'batch size: '+str(batch_size_tst)
        print 'number of batches: '+str(n_test_batches)    

    #Unpickle Models
    if args.v == 1:
         print '... loading pickled dbn.'
    f_dbn = file(model_dir_path+dbn_model, 'rb')
    dbn = cPickle.load(f_dbn)
    f_dbn.close()

    if args.per is not None:
        #Phoneme Error Rate Calculation
        
        if args.v == 1:
            print '... loading pickled HMM.'
        f_hmm = file(model_dir_path+'HMM_final.save', 'rb')
        pi,A = cPickle.load(f_hmm)
        f_hmm.close()
        
        #print pi
        
        if args.v == 1:
            print '... initializing HMM.'
        hmm = HMM(initial=pi, transition=A, emission=dbn, phoneme_priors=prior_tr)
       # print prior_tr
       # print hmm.pi.get_value()
       # print hmm.A.get_value()
        
        viterbi_fn = hmm.build_viterbi_fn(test_set_x, utterances_idces_tst)
        sols = [viterbi_fn(i)[0] for i in xrange( utterances_idces_tst.get_value().shape[0] )]
        per = myutils.compute_per(sols, stacked_labels_tst, frame_idces_tst)
        
        #print np.array(sols[0][4])
        #path_list = path.split('/')
        print 'Phoneme Error Rate in '+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+':   %0.3f' % (per)
    
    #forward
    else:
    
        if args.v == 1:
            print '... computing forward pass.'
        if args.rl:
            f_res = dbn.compute_forward(test_set_x, test_set_y, batch_size_tst, binary=False)
        else:
            f_res = dbn.compute_forward(test_set_x, test_set_y, batch_size_tst)
        
        #Process forward result to have the the format [[frame,rbm_layers,sigm_layers,classified] , ...] 
        if args.v == 1:
            print '... processing forward result.'

        
        data,batch_errors  = myutils.process_forward_data(f_res,test_set_x,test_set_y,batch_size_tst,n_phonemes,len(hidden_layers_sz))
        
        
        print 'Forward Result: %f' % (np.mean(batch_errors))
        
        if args.save:
            if args.v ==1 :
                print '... saving results to file.'
            if args.training:
                if args.rl:
                    f = file(model_dir_path+'Forward_tr_rl.save', 'wb')
                else:
                    f = file(model_dir_path+'Forward_tr.save', 'wb')
            else : 
                if args.rl:
                     f = file(model_dir_path+'Forward_tst_rl.save', 'wb')
                else:
                    f = file(model_dir_path+'Forward_tst.save', 'wb')
                    
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
        

if __name__ == "__main__":
    main()

    
