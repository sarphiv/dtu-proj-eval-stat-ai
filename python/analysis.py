#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

from sklearn.decomposition import PCA



#%%
def get_dataset(file_name, spatial_dimensions = (True, True, True)):
    #Load dataset
    dataset = np.load(file_name)
    
    #Create spatial dimensions mask
    spat_mask = np.array(spatial_dimensions)

    #Reshapes to (1600, 300), 
    # where first dimension is one trajectory,
    # and each 10 trajectories the person is changed
    # and each 100 trajectories the experiment is changed.
    # Where each trajectory is listed as x0, y0, z0, ..., x299, y299, z299
    X = dataset[:, :, :, :, spat_mask].reshape((reduce(np.multiply, dataset.shape[0:3]), -1))
    #One hot encode person
    X_p_idx = np.tile(np.arange(10).repeat(10), 16)
    X_p_data = np.zeros((X.shape[0], dataset.shape[1]))
    X_p_data[np.arange(X.shape[0]), X_p_idx] = 1
    #Prepend person feature onto X
    X = np.hstack((X_p_data, X))


    #One-indexed experiment number labels
    y = (np.arange(16) + 1).repeat(100)


    return (X, y)


#Get dataset
X, y = get_dataset("./data_numpy.npy", spatial_dimensions=(True, True, True))
#Standardize dataset
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

#
pca = PCA()
pca.fit(X_std)


#%%
#Set resolution
plt.rcParams['figure.dpi'] = 400


plt.grid()
plt.plot(np.arange(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_), '.', markersize=4)
plt.xlabel("Number of PC")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Cumulative explained variance")
plt.show()


def plot_in_pc(axis, pc_x, pc_y):
    pca_points = pca.transform(X_std)

    highlighted_class = 16

    for c in range(min(y), max(y) + 1):
        if c == highlighted_class:
            continue

        points = pca_points[y == c, :]
        axis.plot(points[:, pc_x], points[:, pc_y], '8', markersize=2, alpha = 0.5)


    points = pca_points[y == highlighted_class, :]
    axis.plot(points[:, pc_x], points[:, pc_y], '8', markersize=2, alpha = 0.9, color='k')


    axis.set_title(f"PC {pc_y}-{pc_x}")
    axis.set_xlabel("")
    axis.set_ylabel("")
    



fig , axs = plt.subplots(2, 3, figsize = (2*3, 3.6), constrained_layout = True)
fig.suptitle(f"Principal component projections", fontsize = 17)
for i, (pc_x, pc_y) in enumerate([(1, 0), (2, 0), (3, 0), (1, 2), (2, 3), (1, 3)]):
    plot_in_pc(axs[i // 3, i % 3], pc_x, pc_y)


plt.show()
