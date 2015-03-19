#!/bin/env python

import time as t
import myutils
#import DeepLearningTB.DBN as db
import numpy as np
#import theano
#import theano.tensor as T
import cPickle
import argparse
import os

import matplotlib.pyplot as plt
#from  matplotlib.colors import Colormap
import matplotlib.cm as cm
from matplotlib.mlab import PCA

def main():
    parser = argparse.ArgumentParser( description='Parameters Specification.')
    parser.add_argument("-v", help="Show output verbosity.", type=int)
    parser.add_argument("-s", help="Option to specify a number of samples", type=int)
    parser.add_argument("-bl", help="Specify binary layer to study", type=int)
    parser.add_argument("-l", help="Number of hidden layers.", type=str)
    parser.add_argument("-e", help="Pretraining epochs (for getting the model).", type=int)
    parser.add_argument("-tr", help="Number training utterances to take.", type=int)
    parser.add_argument("-mfcc", help="mfcc data.", type=int)
    args = parser.parse_args()

    n_phonemes = 61
    sub_samples = args.s if args.s is not None else 10
    pretraining_epochs = 225 if args.e is None else args.e
    b_layer_idx =  args.bl-1 if args.bl is not None else None
    models_base_path = os.environ['PCODE']+'/Models/'
    
    if args.l is not None :
        hidden_layers_sz = [int(i) for i in args.l.split(',')]
        model_dir_path = models_base_path+str(len(hidden_layers_sz))+'x'+str(hidden_layers_sz[0])+'_pr_'+str(pretraining_epochs)+'/'
    
    if args.v:
        print '... loading forward data.'
    if args.mfcc: 
        X_ = [[]]*n_phonemes
        Y_ = [[]]*n_phonemes
    
        core_test_set = ['mdab0','mwbt0','felc0',
                         'mtas1','mwew0','fpas0',
                         'mjmp0','mlnt0','fpkt0',
                         'mlll0','mtls0','fjlm0',
                         'mbpm0','mklt0','fnlp0',
                         'mcmj0','mjdh0','fmgd0',
                         'mgrt0','mnjm0','fdhc0',
                         'mjln0','mpam0','fmld0']

        mfcc_f_tst = myutils.read_mfcc_files(os.environ['TIMIT']+'/test',spk_set=core_test_set)
        stacked_features_tst,stacked_labels_tst,frame_idces_tst,prior_tst,_,_ = myutils.process_mfcc_files(mfcc_f_tst, n_phonemes)

        for i in xrange(stacked_features_tst.shape[0]):
            X_[stacked_labels_tst[i]].append(stacked_features_tst[i])
            #Y_[stacked_labels_tst[i]].append(stacked_labels_tst[i])
        X = []
        for phn in X_ :
            
            idces = np.random.choice(range(len(phn)),min(sub_samples,len(phn)),replace=False)
            for idx in idces:
                X.append(phn[idx])
                
                
    else:
        if args.tr:
            f = file( model_dir_path+'Forward_tr.save', 'rb')
        else:
            f = file( model_dir_path+'Forward_tst.save', 'rb')
        
        data = cPickle.load(f)
        f.close()
    
    #[[frame,rbm_layers,sigm_layers, output] , ...] 
        X_ = []
        Y_ = []
    #clf = KMeans(n_clusters=n_phonemes,precompute_distances=True,n_jobs=-1)
        for phn_idx in xrange(len(data)):
            phn=data[phn_idx]
            idces = np.random.choice(range(len(phn)),min(sub_samples,len(phn)),replace=False)
        
            for i in idces:           
                if args.bl:
                    X_.append(phn[i][b_layer_idx])
                    Y_.append(phn_idx)
                 
        X = np.vstack(X_)
        Y = np.vstack(Y_)
    
    
    
    #results = PCA(X)

    #cov = np.cov(X.T)
    #print np.sum(results.fracs > 1e-2)




    #plot corr
    corrcoef = np.corrcoef(X)
 
#    figcorr = plt.figure(figsize=(10,10),dpi=100)
    fig = plt.figure(figsize=(10,10),dpi=100) 
    cax = plt.imshow(corrcoef,cmap = cm.binary)
    cbar = fig.colorbar(cax, ticks=[np.min(corrcoef),  np.max(corrcoef)], orientation='vertical')
    cbar.ax.set_yticklabels(['Low', 'High'])# horizontal colorbar
    plt.grid(True)

    phonemes = ['h#','sh','ix','hv','eh','dcl','jh','ih','d',
                'ah','kcl','k','s','ux','q','en','gcl','g',
                'r','w','ao','epi','dx','axr','l','y','uh',
                'n','ae','m','oy','ax','dh','tcl','iy','v',
                'f','t','pcl','ow','hh','ch','bcl','b','aa',
                'em','ng','ay','th','ax-h','ey','p','aw','er',
                'nx','z','el','uw','pau','zh','eng']
    plt.xticks(np.arange(0, args.s*61, args.s), phonemes)
    plt.yticks(np.arange(0, args.s*61, args.s), phonemes)
    
    plt.setp(plt.xticks()[1], rotation=90)
    #plt.setp(plt.yticks()[1], rotation=30)
    
    #plt.annotate('Corrr', xy=(640, 1680), xytext=(580, 1640),
     #       arrowprops=dict(facecolor='black' ,shrink=0.05),
      #      )
    #pylab.xticks([1, 2, 3], ['mon', 'tue', 'wed'])
    plt.show()
    
   
    






if __name__ == "__main__":
    main()


    
