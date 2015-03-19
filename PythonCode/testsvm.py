#!/usr/bin/env python

#import time as t
import myutils
#import DeepLearningTB.DBN as db
import numpy as np
#import theano
#import theano.tensor as T
#from hmm import HMM,EmissionModel
import cPickle
import argparse
import os
from sklearn import svm



def main():
    parser = argparse.ArgumentParser( description='Parameters Specification.')
    parser.add_argument("-v", help="Show output verbosity.", type=int)
    #parser.add_argument("-m", help="Option to specify a model.", type=str)
    parser.add_argument("-C", help="Option to specify a model.", type=float)
    parser.add_argument("-linear", help="Option to specify linear kernel.", type=str)
    #parser.add_argument("-s", help="Option to specify a number of samples", type=int)
    parser.add_argument("-bl", help="Specify binary layer to study", type=int)
    parser.add_argument("-rl", help="Specify sigmoid layer to study", type=int)
    #parser.add_argument("-mfcc", help="Specify MFCC study", type=int)
    parser.add_argument("-tst", help="Number training utterances to take.", type=int)
    parser.add_argument("-l", help="Number of hidden layers.", type=str)
    parser.add_argument("-e", help="Pretraining epochs (for getting the model).", type=int)
    parser.add_argument("-tin", help="Tin dataset.", type=int)

    args = parser.parse_args()
    
    error_cost = 1.0 if args.C is None else args.C
    n_phonemes = 61
    utterances_tst = args.tst if args.tst is not None else None
    models_base_path = os.environ['MT_ROOT']+'/Models/'
    extend_input_n = 11
    pretraining_epochs = 225 if args.e is None else args.e

    if args.l is not None :
        hidden_layers_sz = [int(i) for i in args.l.split(',')]
        if args.tin:
            model_dir_path = models_base_path+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+'_pr_'+str(pretraining_epochs)+'_tin/'
        else:
            model_dir_path = models_base_path+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+'_pr_'+str(pretraining_epochs)+'/'
    else:
        hidden_layers_sz = None
        model_dir_path = models_base_path+'SVM/'
  

    core_test_set = ['mdab0','mwbt0','felc0',
                     'mtas1','mwew0','fpas0',
                     'mjmp0','mlnt0','fpkt0',
                     'mlll0','mtls0','fjlm0',
                     'mbpm0','mklt0','fnlp0',
                     'mcmj0','mjdh0','fmgd0',
                     'mgrt0','mnjm0','fdhc0',
                     'mjln0','mpam0','fmld0']
    
    if args.bl:
        if args.v:
            print '... loading binary vectors.'
            
        f = file(model_dir_path+'/Forward_tst.save', 'rb')
        data = cPickle.load(f)
        f.close()
        X = []
        Y = []
        
        bl_idx = args.bl-1
        for phn_idx in xrange(n_phonemes):
            phn = data[phn_idx]
            for i in xrange(len(phn)):
                X.append(phn[i][bl_idx])
                Y.append(phn_idx)
    
    elif args.rl:
        if args.v:
            print '... loading real sigmoid vectors.'
            
        f = file(model_dir_path+'/Forward_tst_rl.save', 'rb')
        data = cPickle.load(f)
        f.close()
        X = []
        Y = []
        
        rl_idx = args.rl-1
        for phn_idx in xrange(n_phonemes):
            phn = data[phn_idx]
            for i in xrange(len(phn)):
                X.append(phn[i][rl_idx])
                Y.append(phn_idx)
        
    else:

        if args.v == 1:
            print '... loading pickled mean and std.'
        
        f = file(model_dir_path+'mean_std_pr.save', 'rb')

        mean_tr,std_tr,prior_tr = cPickle.load(f)
        f.close()
        
        if args.v >= 1:
            print '... loading core test.'
        mfcc_f_tst = myutils.read_mfcc_files(os.environ['TIMIT']+'/test',spk_set=core_test_set)
        stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes, n=utterances_tst,mean=mean_tr,std=std_tr)

    
        stacked_features_tst = myutils.extend_input(stacked_features_tst, frame_idces_tst, extend_input_n)
        
        X = stacked_features_tst 
        Y = stacked_labels_tst
        
        if args.v >= 1 :
            print '====Training===='
            print 'Total utterances: ' + str(frame_idces_tst.shape[0])
            print 'Total MFCCs: ' + str(stacked_features_tst.shape[0])
        
    ##### Load Svm Classifier
    if args.linear is not None:
        
        if args.bl:
            path = model_dir_path+'LinearSVM_bl_'+str(args.bl)+'.save'
      #      f = file(path , 'rb')
        elif args.rl:
            path = model_dir_path+'LinearSVM_rl_'+str(args.rl)+'.save'
        else:
            path = model_dir_path+'LinearSVM.save'
     #       f = file(path, 'rb')
         
        
    else:
        if args.bl:
            path = model_dir_path+'NonLinearSVM_bl_'+str(args.bl)+'.save'
    #        f = file(path, 'rb')
        elif args.rl:
            path = model_dir_path+'NonLinearSVM_rl_'+str(args.rl)+'.save'
            
        else:
            path = model_dir_path+'NonLinearSVM.save'
            
    f = file(path, 'rb')
            
    if args.v:
       if args.v:
            print '... loading classifier path: '+path
    
    clf = cPickle.load(f)
    f.close()
        
        
    if args.v:
        print '... computing score.'
        
    score = clf.score(X,Y)
    path_list = path.split('/')
    print 'Score for '+path_list[-2]+'_'+path_list[-1]+': %f'%((1-score)*100.)
    

    
    
if __name__ == "__main__":
    main()


    
