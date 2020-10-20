"""descibision of a function

Parameters
----------
arg1 : ((type))
    descibision
arg2 : (type)
    descibision

Return
------
arg1 : (type)
    descibision
"""
"""
"""
from copy import deepcopy
import random
import numpy as np
# import gaussian_tool as gt
from buffer import Exp_Buffer
# from region import Region_Cluster
from collections import OrderedDict
from qmodel import Value_Function

import torch
import torch.nn.functional as F
import torch.optim as optim
from region import Region_Cluster
from scipy.special import softmax
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import ipdb

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class Agent():

    def __init__(self, init_Q_value=0 ,seed = 0, dim =2 , batch_size = 2, lr = 0.1 , gamma=1,tau = 1e-2, if_soft_update =True  ):

        """
        param
        -----
        seed(int): random seed
        dim(int): the dimension of position space (not include the contact dim.)
        batch_size(int):
        lr(float):learning rate
        gamme(float):the discount factor

        """
        # ordered_sample to determine if sequentially draw the sample to update or not
        self.batch_size = batch_size
        self.ordered_sample = False

        self.lr = lr
        self.demo_acts_dict = OrderedDict()
        self.num_of_demo_goal = 0
        self.demo_goal = []
        self.gamma = gamma
        self.tau = tau
        self.if_soft_update = if_soft_update
        self.init_Q_value = init_Q_value
        self.memory = Exp_Buffer(batch_size= self.batch_size)
        self.regions_dict = OrderedDict()
        # The dimension of the workspace (2 or 3).
        self.dim = dim
        self.final_goal_state = 0
        """
        self.act for new skills
        """
        self.actions_dict = OrderedDict()
        # the state dim contain the position and contact
        self.state_size = (dim+dim*2)
        self.action_size = 0
        self.seed = random.seed(seed)

        # Funnel operation variables
        self.funnels = Region_Cluster(dim=self.dim)
        self.funnels_inf_list=[]
        self.funnels_amount= 0

    def demo_record(self,goal_tuples):
        """To record a task demonstration skill with several goal states,
        i.e. the goal 3D position of each initial skill.
        Use ROS interface to get the position information of robot.

        Parameters:
            goal_tuples(list):a list of 2 or 3 dim array.
        """
        self.demo_goal = goal_tuples
        self.demo_amount = len(goal_tuples)
        for i, goal in enumerate(goal_tuples):
            action_index = str(i)
            self.demo_acts_dict[action_index] = goal
        self.final_goal_state = goal_tuples[-1]

        print("Agent: agent has recorded the demonstration as its original skills: {} \n".format(self.demo_acts_dict))
        print("Agent: agent  recorded the final goal state as : {} \n".format(self.final_goal_state))
        self.action_size = len(self.demo_acts_dict)

        return self.demo_acts_dict

    def get_demo_acts_dict(self):
        """Return the action dictionary of demonstration
        Return
        ----------
        demo_acts_dict:(dict)
            keys(string): index 
            values(tuple): goal position
        """
        self.demo_acts_dict
        return self.demo_acts_dict

    def exp_record(self,episode_list):
        """Record experience tuples of an episode to experience list
        Param
        ----------
        episode_list:(list)
            a list of experience tuple.
        """
        for exp_tuple in episode_list:
            state, action, reward, next_state, is_goal = exp_tuple
            self.memory.add(state, action, reward, next_state, is_goal)
    
    # return a list of namedtuple for experience.
    def get_exp_list(self):
        """Return the list of all the restored experience tuples
        Return
        ----------
        experiences_list:(list)
            a list of experience tuple.
        """
        experiences_list = self.memory.get_experience_list()
        return experiences_list

    def learn_funnels_infs(self):
        experiences_list = self.memory.get_experience_list()
        self.funnels_inf_list = self.funnels.learn_funnels(experiences_list, self.demo_acts_dict)
        for funnel in self.funnels_inf_list:
            son_list = []
            father_list = []
            funnel.append(son_list)
            funnel.append(father_list)
            
    def get_funnels_amount(self):
        """Get the amount of funnels.
        Return
        ------
        self.funnels_amount(int):
        """
        pass
        return self.funnels_amount



    def learn_funnels_directed_graph(self,directed_graph_threshold, clip_threshold =1e-15 ):
        """Construct region directed graph used for algorithm 2.
        """
        # Clip the probability, to avoid the final funnel have wierd son funnel.

        exps_list = self.memory.get_experience_list()
        for i,a_i, r_i, i_son, _ in self.funnels_inf_list:
            score_i_list=[]
            for j, a_j ,r_j, _, _ in self.funnels_inf_list:
                score_i_j = 0
                # Here will inlude the score_i_i = 0 , donnot know whether this will affect the result.
                if i != j:
                    # Integrate all the exp tuples
                    for exp in exps_list:
                        phi_i = multivariate_normal.pdf(exp.state, r_i.values()[0][0], r_i.values()[0][1])
                        phi_i = phi_i if phi_i > clip_threshold else 0

                        psi_j = multivariate_normal.pdf(exp.next_state, r_j.values()[0][0], r_j.values()[0][1])
                        psi_j = psi_j if psi_j > clip_threshold else 0

                        score_i_j += phi_i * psi_j                # multiply 1e+3 to avoid vanishing.
                score_i_list.append(score_i_j)

            denominator = sum(score_i_list)
            # comput the transition probability
            i_probs = score_i_list / denominator if denominator != 0 else np.zeros(len(score_i_list))
            # construct the directed relationship
            for j, a_j ,r_j, j_son, j_father  in self.funnels_inf_list:
                if i_probs[j] > directed_graph_threshold:
                    i_son.append(j)
                    j_father.append(i)
        print(self.funnels_inf_list)

    def init_value_function(self):
        """Initialization of the approximate value function with the amount of funnels.
        """
        self.funnels_amount = len(self.funnels_inf_list)

        self.qlearning_method = Value_Function(lr=self.lr, demo_acts_dict = self.demo_acts_dict, \
                                        funnels_inf_list = self.funnels_inf_list,gamma = self.gamma,\
                                        init_Q_value = self.init_Q_value, if_soft_update = self.if_soft_update,tau = self.tau)



    def learn_initial_policy(self):
        """Learn the policy with the original experience tuple.
        """
        experiences = self.memory.batch_sample()
        # step
        loss = self.qlearning_method.q_learn(experiences)
        self.qlearning_method.soft_update()
        return loss

    def choose_act(self,state):
        picked_action_dict = {}
        # Forward the state to get all action Q
        Q_s_tensor = self.qlearning_method.forward(state)
        # Convert to probability using softmax
        action_prob_vector = softmax(Q_s_tensor)
        # Pick the action under that probability
        a_index = np.arange(self.action_size)
        picked_a_index = np.random.choice( a_index, p = action_prob_vector)
        # Return the picked action dict
        picked_action_dict[str(picked_a_index)] = self.demo_acts_dict[str(picked_a_index)]
        return picked_action_dict

    def get_funnels_q_value(self):
        return self.qlearning_method.get_param()

    def plot_funnels_points(self):
        exps_list = self.memory.get_experience_list()
        # Only need to init subplot one time.
        figsize = 20,8
        figure, ax = plt.subplots(figsize=figsize)
        coArray=['g','m','c','gold',"r",'b','w','k']
        shArray = ["D","o","p","s",'v',"X"]
        for i,a_i, r_i, _, _ in self.funnels_inf_list:
            funnel_points_x_list = []
            funnel_points_y_list = []
            for exp in exps_list:
                # Plot each tuple transition line
                tuple_transition_line_x = []
                tuple_transition_line_y = []
                tuple_transition_line_x.append(exp.state[0])
                tuple_transition_line_x.append(exp.next_state[0])
                tuple_transition_line_y.append(exp.state[1]) 
                tuple_transition_line_y.append(exp.next_state[1])
                plt.plot(tuple_transition_line_x, tuple_transition_line_y ,color='gray', linewidth=0.4 , alpha =0.05 ) 
                # Plot each funnel ponits
                if exp.action == a_i and  multivariate_normal.pdf(exp.state, r_i.values()[0][0], r_i.values()[0][1]) > 1e-5:
                    funnel_points_x_list.append(exp.state[0])
                    funnel_points_y_list.append(exp.state[1])
            # Pop out different shape and color for plot
            colour =  coArray.pop(0)
            shape =  shArray.pop(0)
            plt.scatter(funnel_points_x_list, funnel_points_y_list, color=colour, marker = shape, s= 40,linewidths=[0.5], edgecolors='black' )
        # Draw the terminate point with 'done' = True
        final_region_point_x = []
        final_region_point_y = []
        for exp in exps_list:    
            if exp.done == True:
                final_region_point_x.append(exp.next_state[0])
                final_region_point_y.append(exp.next_state[1])
        plt.scatter(final_region_point_x, final_region_point_y, color='black', marker = "X", s= 100,alpha = 0.3, linewidths=[0.5], edgecolors='black' )
        # Show all the plot.
        plt.ylabel('y')
        plt.xlabel('x')
        plt.savefig('transition_plot.pdf')
        plt.show()

    def plot_funnels_directed_graph(self):
        figsize = 20,8
        figure, ax = plt.subplots(figsize=figsize)
        coArray=['g','m','c','gold',"r",'b','w','k']
        # First, draw all the funnels regions.
        for i,a_i, r_i, i_sons, _ in self.funnels_inf_list:
            x = r_i.values()[0][0][0]
            y = r_i.values()[0][0][1]
            colour =  coArray.pop(0)
            plt.scatter(x, y, color=colour, s= 6000,alpha = 0.4, linewidths=[1], edgecolors='black' )
        colour =  coArray.pop(0)
        plt.scatter(30, 0, color=colour, s= 1,alpha = 0.001, linewidths=[1], edgecolors='black' )

        # Second, Draw all the arrow to represent the directed graph.
        coArray=['g','m','c','gold',"r",'b','w','k']
        shift = -0.5
        for i,a_i, r_i, i_sons, _ in self.funnels_inf_list:
            father_x = r_i.values()[0][0][0]
            father_y = r_i.values()[0][0][1]
            colour =  coArray.pop(0)
            for son_index in i_sons:
                son_x = self.funnels_inf_list[son_index][2].values()[0][0][0]
                son_y = self.funnels_inf_list[son_index][2].values()[0][0][1]
                plt.annotate('.',xy=(son_x,son_y),xytext=(father_x,father_y),arrowprops=dict(alpha= 0.4,facecolor=colour,arrowstyle="fancy,head_length=1.5,head_width=1.5,tail_width=1",connectionstyle="angle3,angleA=0,angleB=-90"))
                plt.text(father_x+(son_x - father_x)/2 +shift,father_y+(son_y - father_y)/2+shift,'funnel {}'.format(i),fontsize=15)
                shift = -shift

        # Compute the finnal region (manually compute for a specific situation, should ask Jim to compute the mean of the final region)
        son_index = self.funnels_inf_list[1][3][0]
        father_x = self.funnels_inf_list[son_index][2].values()[0][0][0]
        father_y = self.funnels_inf_list[son_index][2].values()[0][0][1]
        plt.annotate('.',xy=(30,0),xytext=(father_x,father_y),arrowprops=dict(alpha= 0.4,facecolor='gold',arrowstyle="fancy,head_length=1.5,head_width=1.5,tail_width=1",connectionstyle="angle3,angleA=0,angleB=-90"))
        plt.text(25 +shift,0+shift,'funnel {}'.format(3),fontsize=15)
        plt.title(r'Directed Graph of Funnels',fontproperties='SimHei',fontsize=20)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.savefig('directed_graph_plot.pdf')
        plt.show()
