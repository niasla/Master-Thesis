#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np
def main():
    
    units = [512,1024,2048]
    
    per2 = 100*np.array([0.453, 0.455, 0.469])
    per3 = 100*np.array([0.457, 0.467, 0.469])
    per4 = 100*np.array([0.450, 0.464, 0.485])

    
    fig=plt.figure(figsize=(7,5),dpi=100)  

    line1,line2,line3 = plt.plot(units,per2,'ro-',
                                 units,per3,'b-+',
                                 units,per4,'g-x')
    
    all_v = np.concatenate((per2, per3, per4))
    x_range = max(units)-min(units)
    y_range = np.floor(np.max(all_v) - np.min(all_v))
    
    xmax = max(units) + x_range * 0.1
    xmin = min(units) #- x_range * 0.1
    ymax = np.max(all_v) + y_range*0.1
    ymin = np.min(all_v) - y_range*0.1
    
    plt.axis([ xmin, xmax, ymin, ymax])
    
    y_step = 0.4
    x_step = 512

    plt.xticks([512, 1024, 2048])
    plt.yticks(np.arange(np.min(all_v),np.max(all_v),y_step))




    plt.legend( (line1, line2, line3), ('2 Layers', '3 Layers', '4 Layers'),loc=2 )
    

    plt.title('Effect of varying layer size on PER ')
    plt.ylabel('Phoneme Error Rate (PER) %')
    plt.xlabel('Layer Size')


    plt.show()


if __name__ == "__main__":
    main()
