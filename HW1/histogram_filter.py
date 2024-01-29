import numpy as np

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        #
        #Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        #belief distribution according to the Bayes Filter.
        #:param cmap: The binary NxM colormap known to the robot.
        #:param belief: An NxM numpy ndarray representing the prior belief.
        #:param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        #:param observation: The observation from the color sensor. [0 or 1].
        #:return: The posterior distribution.
        #

        ### Your Algorithm goes Below.
        cmap = np.rot90(cmap,-1)
        belief = np.rot90(belief,-1)
        N_dimension = cmap.shape[0]
        M_dimension = cmap.shape[1]
        sensor_ob_true,sensor_ob_false=0.9,0.1 # probabilities for sensor accuracy 
        execute_move,execute_stay=0.9,0.1      # probabilities for movement execution
        
        n_belief=np.zeros([N_dimension,M_dimension])
        sensor_prob = np.zeros([N_dimension,M_dimension])
        
        """
        action: -1 move backward; 0 stay; 1 move forward       
        """ 
        # Update belief based on the action
        for step_x in range(N_dimension):
            for step_y in range(M_dimension):
                if  step_x+action[0]<N_dimension and step_x+action[0]>=0 and step_y+action[1]<M_dimension and step_y+action[1]>=0:
                    #For robots not on the boundaries 
                    n_belief[step_x+action[0]][step_y+action[1]] += belief[step_x][step_y]*execute_move
                    n_belief[step_x][step_y] += belief[step_x][step_y]*execute_stay
                else:
                    #For robots on the boundaries, not move 
                    n_belief[step_x][step_y] += belief[step_x][step_y]
        
        # Update belief based on the sensor observation
        for step_x in range(N_dimension):
            for step_y in range(M_dimension):
                bool_ = int(cmap[step_x][step_y] == observation)
                sensor_prob[step_x][step_y] = bool_ * sensor_ob_true + (1 - bool_) * sensor_ob_false
        
        n_belief = np.multiply(sensor_prob, n_belief)
        n_belief /= sensor_prob.sum()                  # Normalize the distribution
        
        
        # Find the position with the highest belief for test
        tmp = 0
        for i in range(N_dimension):
            for j in range(M_dimension):
                if n_belief[i][j] > tmp:
                    tmp = n_belief[i][j]
                    state = [i,j]
        
        return np.rot90(n_belief,k=1)
    


        
        

        
        
        
