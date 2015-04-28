#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():

    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,5),dpi=100)  
    

    #3 Layers

  #  plt.subplot(211)
    
    
    unit512  =[53.748510, 54.475588, 51.349671]
    
    unit1024 = [56.485847, 54.862442, 51.423933]

    unit2048 = [58.340673, 54.458318, 50.995631]

  

    blayers = [1,2,3]
    
    line1,line2,line3 = plt.plot(blayers,unit512,'rx-',
                                 blayers,unit1024,'bo--',
                                 blayers,unit2048,'g+-.')
                                       
   
    all_v = np.concatenate((unit512, unit1024, unit2048))
    
    x_range = max(blayers)-min(blayers)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    xmax = max(blayers)  + x_range * 0.1
    xmin = min(blayers)  - x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([xmin, xmax, ymin, ymax])
    
    plt.axhline(y=56.83, xmin=0, xmax=xmax-0.3,color='k',linestyle='--')
    plt.text(1.7,57,'MFCC error rate')

    y_step = 1
    x_step = 1

    plt.xticks(np.arange(min(blayers), max(blayers)+1, x_step))
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),y_step))

    plt.xlabel('Binary Hidden Layer Level')
    plt.ylabel('Test Set Error (%)')
    #plt.title('SVM core test set error - 3 layers architecture.')
    plt.legend( (line1, line2, line3), ('512 Units', 
                                        '1024 Units', 
                                        '2048 Units'), prop={'size':12})
    
    
       

    plt.show()           

if __name__ == "__main__":
    main()
