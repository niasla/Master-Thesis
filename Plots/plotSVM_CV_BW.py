#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():

    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,5),dpi=100)  
    

    #3 Layers

    #plt.subplot(211)
    
    
    
    C1000   = np.array([ 0.42366076,  0.43833814,  0.4189343,   0.42926506,  0.43211962] )*100
    C1      = np.array([ 0.42166076,  0.42933814,  0.4189343,   0.43926506,  0.43211962] )*100
    C10     = np.array([ 0.42466076,  0.4233814,  0.4189343,   0.43926506,  0.43211962] )*100
    C100    = np.array([ 0.42166076,  0.42933814,  0.4159343,   0.43926506,  0.43511962] )*100

    x = [C1, C10, C100, C1000]

    C = [1, 10, 100, 1000]
    
    plt.boxplot(x)
                                       
  
    all_v = np.concatenate((C1, C10, C100, C1000))
    
    x_range = max(C)-min(C)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    xmax = max(C)  + x_range * 0.1
    xmin = min(C)  - x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
   
   # plt.axis([xmin, xmax, ymin, ymax])
    
    y_step = 0.5
    x_step = 1

   # plt.xticks(np.arange(min(C), max(C)+1, x_step))
    plt.xticks([1, 2, 3, 4], [1, 10, 100, 1000])
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),y_step))

    plt.xlabel('Error Cost (C)')
    plt.ylabel('Cross Validation Error (%)')
    #plt.title('SVM Cross Validation for MFCC features.')
   
    

    plt.show()           

if __name__ == "__main__":
    main()
