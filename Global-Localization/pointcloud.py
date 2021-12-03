import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import hdbscan
import time
import seaborn as sns

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 1, 's' : 10, 'linewidths':0}

def plot_clusters(data, labels):
    tempData = np.delete(data, np.where(labels == -1),axis= 0)
    tempLabels = np.delete(labels, np.where(labels == -1))
    palette = sns.color_palette('deep', np.unique(tempLabels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in tempLabels]
    print(tempData)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = list(p[0] for p in tempData)
    y = list(p[1] for p in tempData)
    z = list(p[2] for p in tempData)
    ax.scatter(x, y, z, c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by HDBSCAN', fontsize=24)
    plt.show()

pcd = o3d.io.read_point_cloud("D:/Data/Uni/Mobile Robotik/code/Gesture-Recognition/Global-Localization/Room.PLY")
#o3d.visualization.draw_geometries([pcd])


abb = pcd.get_axis_aligned_bounding_box() #get bounding box
pts = np.asarray(abb.get_box_points()) #extract all 8 points
top = 1
bot = -0.45
#print(pts)
#get min and max of z axis
minZ = round(np.amin(pts, axis=0)[2],6)
maxZ = round(np.amax(pts, axis=0)[2],6)
#change z axis to desired values for cropping
for v in pts:
    if round(v[2],6) == minZ:
        v[2] = bot
    elif round(v[2], 6) == maxZ:
        v[2] = top

vec= o3d.cpu.pybind.utility.Vector3dVector(pts) #create vector object for sliced bounding box
obb = o3d.geometry.OrientedBoundingBox()
obb = obb.create_from_points(vec) #create sliced bounding box

cpcd = pcd.crop(obb) #crop the pointcloud

cl, ind = cpcd.remove_radius_outlier(nb_points=30, radius=0.10) #remove outliers
#remove points that would not be visible anyways
#diameter = np.linalg.norm(np.asarray(cpcd.get_max_bound()) - np.asarray(cpcd.get_min_bound()))
#_, pt_map = cpcd.hidden_point_removal([0,0,diameter], diameter*100)
#o3d.visualization.draw_geometries([pcd.select_by_index(ind)]) #visualize

# Clustering
data = np.asarray(cpcd.points)
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40,
    metric='euclidean', min_cluster_size=10, min_samples=20, p=None)
clusterer.fit(data) # fit the cropped pointcloud
print(data.size)
print(max(clusterer.labels_))
print(clusterer.labels_.size)
plot_clusters(data,clusterer.labels_)

#newCloud = o3d.geometry.PointCloud()
#newCloud.points = o3d.cpu.pybind.utility.Vector3dVector(cpcd)
#o3d.visualization.draw_geometries([cpcd]) #visualize

