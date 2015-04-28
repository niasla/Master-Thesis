#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():
    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,5),dpi=100)  
    l3 = np.loadtxt(base_path+'4x512_pr_225_dev_err_preeffect.txt',dtype=np.float32)
    l4 = np.loadtxt(base_path+'4x512_pr_0_dev_err_preeffect.txt',dtype=np.float32)
    l3_sz = l3.shape[0]
    l4_sz = l4.shape[0]

    line1,line2 = plt.plot(range(l3_sz),l3,'rx-',
                           range(l4_sz),l4,'bo--')
  
    all_v = np.concatenate((l3,l4))
    
    x_range = np.floor(np.abs(l3_sz-l4_sz))
    y_range = np.floor(np.max(all_v) - np.min(all_v))

    xmax = np.max([l3_sz,l4_sz])+x_range * 0.1
    xmin = 0
    ymax = np.min(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ 0, xmax, ymin, ymax])
    
    plt.xticks(np.arange(0, np.max([l3_sz,l4_sz]), 5))
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),5))

    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Error (%)')
    #plt.title('Validation error with and without pre-training.')
    plt.legend( (line1, line2), ('with pretraining', 'w/o pretraining'))

    plt.show()



if __name__ == "__main__":
    main()
