#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():
    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(5,5),dpi=100)  
    bl1 = [55.335648, 56.510025, 56.731085]  #4x512...
    
    bl2 = [59.414884, 57.737941, 55.689688]

    bl3 = [55.202667, 55.394366, 54.287343]

    bl4 = [50.546604, 50.344542, 49.249607]

    units = [512,1024,2048]
    
    line1,line2,line3,line4 = plt.plot(units,bl1,'rx-',
                                       units,bl2,'bo--',
                                       units,bl3,'g+-.',
                                       units,bl4,'kx:',)
  
    all_v = np.concatenate((bl1,bl2,bl3,bl4))
    
    x_range = max(units)-min(units)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    xmax = max(units)  + x_range * 0.1
    xmin = min(units)  - x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ xmin, xmax, ymin, ymax])
    
    y_step = 1.0
    x_step = 512

    plt.xticks(np.arange(min(units), max(units)+1, x_step))
    plt.yticks(np.arange(np.min(all_v),np.max(all_v),y_step))

    plt.xlabel('Number of Units')
    plt.ylabel('Test-Set Error (%)')
    plt.title('Test set error of different unit numbers in four layer architecture.')
    plt.legend( (line1, line2, line3, line4), ('Binary Layer Level 1', 
                                               'Binary Layer Level 2', 
                                               'Binary Layer Level 3',
                                               'Binary Layer Level 4'))
    line1.set_antialiased(True)
    line2.set_antialiased(True)

    plt.show()



if __name__ == "__main__":
    main()
