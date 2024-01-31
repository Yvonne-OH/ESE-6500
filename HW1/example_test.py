import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']
    

    print("belief_states: \n", belief_states)
    print(belief_states.shape)


    #### Test your code here
        
    belief  = np.ones((20,20))/400
    
    H_filter = HistogramFilter()
    
    for i in range(len(belief_states)):
    
        p = H_filter.histogram_filter(cmap, belief, actions[i], observations[i])
        #belief = p[0]
        #print (p[1])
