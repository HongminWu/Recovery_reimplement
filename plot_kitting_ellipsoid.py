from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import numpy as np
from collections import namedtuple
import region
from collections import OrderedDict
from matplotlib import cm

demo_act_dict = OrderedDict()
goal_tuples = np.array([-1,-1,-1])
demo_act_dict[3]=goal_tuples
demo_act_dict[4]=goal_tuples
demo_act_dict[5]=goal_tuples
demo_act_dict[7]=goal_tuples
demo_act_dict[8]=goal_tuples
demo_act_dict[9]=goal_tuples

exp_tuple_test = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/experience_tuple_no_recovery_skill_positions.npy')
experience_tuple = []
experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
    
for i in range(len(exp_tuple_test)):
    action = OrderedDict()

    state = exp_tuple_test[i][0]
    action[exp_tuple_test[i][1]] = goal_tuples
    reward = exp_tuple_test[i][2]
    next_state = exp_tuple_test[i][3]
    done = exp_tuple_test[i][4]
    e =  experience(state, action, reward, next_state, done)
    experience_tuple.append(e)


# for e in exp_tuple_test:
#     print(e.state)

experience_list = experience_tuple
action_dict = demo_act_dict
# print(action_dict)
# print(experience_list)

obj = region.Region_Cluster()
r_s = obj.learn_funnels(experience_list, action_dict)

# for j in range(len(r_s)):

fig = plt.figure()
ax = fig.gca(projection='3d', adjustable='box')
for i, r_ss in enumerate(r_s):
    print i
    # Radii corresponding to the coefficients:
    a, b = np.linalg.eig(r_ss[2][i][1])
    rx= b[0][0]
    ry= b[1][1]
    rz= b[2][2]

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = np.mat(rx * np.outer(np.cos(u), np.sin(v)))+r_ss[2][i][0][0]
    y = np.mat(ry * np.outer(np.sin(u), np.sin(v)))+r_ss[2][i][0][1]
    z = np.mat(rz * np.outer(np.ones_like(u), np.cos(v)))+r_ss[2][i][0][2]

    # Plot the surface
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.5)

    cset = ax.contourf(x, y, z, zdir='x', offset=-2*rx, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, z, zdir='y', offset=1.8*ry, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, z, zdir='z', offset=-2*rz, cmap=cm.coolwarm)
    max_radius = max(rx, ry, rz)

# ax.set_xlabel('X')
# ax.set_xlim(-2*max_radius, 2*max_radius)
# ax.set_ylabel('Y')
# ax.set_ylim(-1.8*max_radius, 1.8*max_radius)
# ax.set_zlabel('Z')
# ax.set_zlim(-2*max_radius, 2*max_radius)
plt.show()


