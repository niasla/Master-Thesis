#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():
    base_path = os.environ['PCODE'] + '/Results/'
    
    #epochs vs err 4x1024 , 3x1024 
    fig=plt.figure(figsize=(7,5),dpi=100)  
    l2 = np.loadtxt(base_path+'2x512_pr_225_dev_err.txt',dtype=np.float32)
    l3 = np.loadtxt(base_path+'3x512_pr_225_dev_err.txt',dtype=np.float32)
    l4 = np.loadtxt(base_path+'4x512_pr_225_dev_err.txt',dtype=np.float32)
    l5 = np.loadtxt(base_path+'5x512_pr_225_dev_err.txt',dtype=np.float32)
    
    l2_sz = l2.shape[0]
    l3_sz = l3.shape[0]
    l4_sz = l4.shape[0]
    l5_sz = l5.shape[0]

    line1,line2,line3,line4 = plt.plot(range(l2_sz),l2,'rx-',
                           range(l3_sz),l3,'bo--',
                           range(l4_sz),l4,'g+-.',
                           range(l5_sz),l5,'kx-',)
  
    all_v = np.concatenate((l2,l3,l4,l5))
    
    x_range = np.max([l2_sz,l3_sz,l4_sz,l5_sz]) - np.min([l2_sz,l3_sz,l4_sz,l5_sz])
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    xmax = np.max([l2_sz,l3_sz,l4_sz,l5_sz]) + x_range * 0.1
    xmin = 0
    ymax = np.min(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ 0, xmax, ymin, ymax])
    
    plt.xticks(np.arange(0, np.max([l3_sz,l4_sz]), 10))
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),5))

    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Error (%)')
    #plt.title('Validation error evolution in different layers.')
    plt.legend( (line1, line2, line3, line4), ('2x512', '3x512', '4x512','5x512'))

    plt.show()



if __name__ == "__main__":
    main()
