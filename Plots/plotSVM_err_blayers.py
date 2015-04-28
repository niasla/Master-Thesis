#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():

    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,5),dpi=100)  
    

    #3 Layers

    plt.subplot(211)
    
    
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
    
    y_step = 1.0
    x_step = 1

    plt.xticks(np.arange(min(blayers), max(blayers)+1, x_step))
    plt.yticks(np.arange(np.min(all_v),np.max(all_v),y_step))

    plt.xlabel('Layer Level')
    plt.ylabel('Test-Set Error (%)')
    plt.title('SVM Test set error of different layer levels in three and four layer architecture.')
    plt.legend( (line1, line2, line3), ('512 Units', 
                                        '1024 Units', 
                                        '2048 Units'))
    
    
    
    # 4 layers
    
    plt.subplot(212)

    unit512 = [55.335648, 59.414884, 55.202667, 50.546604]
         
    unit1024 = [56.510025, 57.737941, 55.394366, 50.344542]

    unit2048 = [56.731085, 55.689688, 54.287343, 49.249607]

    blayers = [1,2,3,4]
    
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
    
    y_step = 1.0
    x_step = 1

    plt.xticks(np.arange(min(blayers), max(blayers)+1, x_step))
    plt.yticks(np.arange(np.min(all_v),np.max(all_v),y_step))

    plt.xlabel('Number of layer')
    plt.ylabel('Test-Set Error (%)')
    #plt.title('SVM Test set error of different layer levels in three and four layer architecture.')
    plt.legend( (line1, line2, line3), ('512 Units', 
                                        '1024 Units', 
                                        '2048 Units'))
    

    plt.show()           

if __name__ == "__main__":
    main()
