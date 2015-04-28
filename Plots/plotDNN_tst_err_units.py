#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():
    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(5,5),dpi=100)  
    l3  = 100*np.array([0.410203278065 ,0.411550343037, 0.41557431221]) 
    l4 = 100*np.array([0.425280213356, 0.413899093866, 0.4])
   
    units = [512,1024,2048]
    
    line1,line2 = plt.plot(units,l3,'rx-',
                           units,l4,'bo--')
                                 
  
    all_v = np.concatenate((l3,l4))
    
    x_range = max(units)-min(units)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    
    xmax = max(units) + x_range * 0.1
    xmin = min(units) - x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ xmin, xmax, ymin, ymax])
    
    y_step = 0.2
    x_step = 512

    plt.xticks([512, 1024, 2048])
    plt.yticks(np.arange(np.min(all_v),np.max(all_v),y_step))

    plt.xlabel('Number of Layers')
    plt.ylabel('Test-Set Error (%)')
    #plt.title('Test set error .')
    plt.legend( (line1, line2), ('3 Layers architecture', '4 Layers architecture'))
    line1.set_antialiased(True)
    line2.set_antialiased(True)

    plt.show()



if __name__ == "__main__":
    main()
