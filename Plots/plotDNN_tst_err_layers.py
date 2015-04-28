#!/bin/env python
import matplotlib.pyplot as plt
import os
import numpy as np
def main():
    base_path = os.environ['PCODE'] + '/Results/'
    
    
    fig=plt.figure(figsize=(7,5),dpi=100)  






    units512  = 100*np.array([0.400826, 0.425280 , 0.403347, 0.423864 ])
    units1024 = 100*np.array([0.401620, 0.411550 , 0.413899, 0.433363])
    units2048 = 100*np.array([0.418009, 0.415574 , 0.432758, 0.451168])

    layers = [2,3,4,5]

    line1,line2,line3 = plt.plot(layers,units512,'rx-',
                                 layers,units1024,'bo--',
                                 layers,units2048,'g+-.')
  
    all_v = np.concatenate((units512,units1024,units2048))
    
    x_range = 2
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    xmax = 5 +x_range * 0.1
    xmin = 2
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ xmin, xmax, ymin, ymax])
    
    y_step = 1
    x_step = 1

    plt.xticks(np.arange(xmin, max(layers)+1, x_step))
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),y_step))
    
    #Tin's result
    #plt.axhline(y=0, xmin=0, xmax=1)
    
    
    plt.xlabel('Number of Layers')
    plt.ylabel('Test-Set Error (%)')
    #plt.title('Test set error as a function of number of layers.')
    plt.legend( (line1, line2, line3), ('512 Units', '1024 Units', '2048 Units'),loc=2)
    line1.set_antialiased(True)
    line2.set_antialiased(True)

    plt.show()



if __name__ == "__main__":
    main()
