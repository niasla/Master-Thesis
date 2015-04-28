#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():

    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,7),dpi=100)  
    

    #3 Layers

    #plt.subplot(211)
    pr_time = np.array([12542.163434, 22372.0912349, 66130.5016351]) / float(60)

    ft_time = np.array([5143.45696282, 8745.16159797, 23122.959852255]) / float(60)


    all_v = np.vstack([pr_time,ft_time])

    X = np.arange(pr_time.shape[0])

   # n = 12
    #X = np.arange(n)
    #Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    #Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    
    b1 = plt.bar(X, np.floor(ft_time), width = 0.4 ,bottom = pr_time, facecolor='#9999ff', edgecolor='white')
    b2 = plt.bar(X, np.floor(pr_time), width = 0.4 ,facecolor='#ff9999', edgecolor='white')
    
    for x, y in zip(X, ft_time+pr_time):
        plt.text(x + 0.2, y + 0.05, '%d' % np.floor(y), ha='center', va='bottom')
        
    #pl.ylim(-1.25, +1.25)
    
    
    
                                       
  
    #all_v = np.concatenate((C1, C10, C100, C1000))
    
  #  x_range = max(C)-min(C)
  #  y_range = np.floor(np.max(all_v) - np.min(all_v))
  #  xmax = max(C)  + x_range * 0.1
  #  xmin = min(C)  - x_range * 0.1
  #  ymax = np.max(all_v) + y_range*0.1
  #  ymin = np.min(all_v) - y_range*0.1
  # 
  # # plt.axis([xmin, xmax, ymin, ymax])
  #  
    y_step = 200
    #x_step = 100
  #
  # # plt.xticks(np.arange(min(C), max(C)+1, x_step))
    arc = ['5x512','5x1024','5x2048']
    
    #xtickNames = plt.setp(ax1, xticklabels=arc)
    plt.xticks(X, arc)
    #plt.setp(xticksNames, rotation=45, fontsize=8)
    plt.setp(plt.xticks()[1], rotation=30)
    plt.yticks(np.arange(0,1700,y_step))
  #
    plt.xlabel('Architecture')
    plt.ylabel('Time (minutes)')


    plt.legend((b1,b2),('Fine-tuning time','Pre-training time'),loc=2)

  #  #plt.title('SVM Cross Validation for MFCC features.')
   
    

    plt.show()           

if __name__ == "__main__":
    main()