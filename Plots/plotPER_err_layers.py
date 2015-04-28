#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np
def main():
    
  
    fig=plt.figure(figsize=(7,5),dpi=100)


    layers = [2,3,4,5]
   
    unit512  = np.array([45.301 , 45.675, 44.976 , 46.792])
    unit1024 = np.array([45.450 , 46.739, 46.405, 49.456])
    unit2048 = np.array([ 46.937 ,46.921, 48.545, 47.600])

    
    line1,line2,line3 = plt.plot(layers,unit512,'rx-',
                                 layers,unit1024,'bo--',
                                 layers,unit2048,'g+-.')
    
    all_v = np.concatenate((unit512, unit1024, unit2048))
    x_range = max(layers)-min(layers)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    
    xmax = max(layers) + x_range * 0.1
    xmin = min(layers) #- x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ xmin, xmax, ymin, ymax])
    
    y_step = 1
    x_step = 1

    plt.xticks(layers)
    plt.yticks(np.arange(np.floor(np.min(all_v)),np.max(all_v),y_step))




    plt.legend( (line1, line2, line3), ('512 Units', '1024 Units', '2048 Units'),loc=2 )
    

    #plt.title('Effect of varying number of layers on PER ')
    plt.ylabel('Phoneme Error Rate (PER) %')
    plt.xlabel('Number of Layers')





    plt.show()


if __name__ == "__main__":
    main()
