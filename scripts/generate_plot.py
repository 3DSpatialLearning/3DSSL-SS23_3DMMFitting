import numpy as np
import matplotlib.pyplot as plt



# number of views
views = np.array([1, 4, 8, 12])

# mean and standard deviation for views 1, 4, 8, 12
mean_landmark_only = np.array([3.5, 3.5, 3.5, 3.5])
std_landmark_only = np.array([0.5, 0.5, 0.5, 0.5])

mean_landmark_and_rgb = np.array([3.3, 2.8, 2.7, 2.9])
std_landmark_and_rgb = np.array([0.6, 0.29, 0.29, 0.4])

mean_landmark_and_rgbd = np.array([1.7, 1.7, 1.7, 1.7])
std_landmark_and_rgbd = np.array([0.1, 0.1, 0.1, 0.1])

mean_landmark_and_rgbd_chamfer = np.array([1.8, 1.8, 1.8, 1.8])
std_landmark_and_rgbd_chamfer = np.array([0.12, 0.1, 0.11, 0.12])

fig,(ax1)=plt.subplots(1,1)

ax1.errorbar(views, mean_landmark_only, yerr=std_landmark_only, linestyle="dashed", marker='o', markersize=8, label='landmark only')
ax1.errorbar(views, mean_landmark_and_rgb, yerr=std_landmark_and_rgb, linestyle="dashed", marker='o', markersize=8, label='land. + rgb')
ax1.errorbar(views, mean_landmark_and_rgbd, yerr=std_landmark_and_rgbd, linestyle="dashed", marker='o', markersize=8, label='land. + rgbd')
ax1.errorbar(views, mean_landmark_and_rgbd_chamfer, yerr=std_landmark_and_rgbd_chamfer, linestyle="dashed", marker='o', markersize=8, label='land. + rgbd_chamfer')

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.xlabel('number of views')
plt.ylabel('mean error (mm)')
plt.title('Mean error for different number of views')

fig.savefig('comparison.png')