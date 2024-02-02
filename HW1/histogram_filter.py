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
        cmap = cmap
        belief = belief
        N_dimension = cmap.shape[0]
        M_dimension = cmap.shape[1]
        sensor_ob_true,sensor_ob_false= 0.9,0.1 # probabilities for sensor accuracy 
        execute_move,execute_stay= 0.9,0.1      # probabilities for movement execution
        
        n_belief=np.zeros([N_dimension,M_dimension])
        sensor_prob = np.zeros([N_dimension,M_dimension])
        
        """
        action: -1 move backward; 0 stay; 1 move forward       
        """ 
        
        # Update belief based on the action

        
        for step_x in range(N_dimension):
            for step_y in range(M_dimension): 
                if action[0] == 0 and action[1] == 1:  #moving up
                    if step_x == N_dimension - 1:  # bottom
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y]
                    else:  # other
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y] +  execute_move * belief[step_x + 1][step_y]
                
                if action[0] == 0 and action[1] == -1:  # down
                    if step_x == 0:  # top
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y]
                    else:  # other
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y] +  execute_move * belief[step_x - 1][step_y]
            
                if action[0] == 1 and action[1] == 0:  # right
                    if step_y == 0:  # left edge
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y]
                    else:  # other
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y] +  execute_move * belief[step_x][step_y - 1]
            
                if action[0] == -1 and action[1] == 0:  # left
                    if step_y == M_dimension - 1:  # right edge
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y]
                    else:  # other
                        n_belief[step_x][step_y] = execute_stay * belief[step_x][step_y] +  execute_move * belief[step_x][step_y + 1]
                    

        # Update belief based on the sensor observation
        
        for step_x in range(N_dimension):
            for step_y in range(M_dimension):
                bool_ = int(cmap[step_x][step_y] == observation)
                sensor_prob[step_x][step_y] = bool_ * sensor_ob_true + (1 - bool_) * sensor_ob_false
        
        n_belief = np.multiply(sensor_prob, n_belief)
        n_belief /= n_belief.sum()                  # Normalize the distribution
        
        # Find the position with the highest belief for test
        tmp = 0
        for i in range(N_dimension):
            for j in range(M_dimension):
                if n_belief[step_x][step_y] > tmp:
                    tmp = n_belief[step_x][step_y]
                    state = [i,j]
        
        return n_belief
    


        
        

        
        
        
