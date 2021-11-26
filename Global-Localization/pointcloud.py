import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np

pcd = o3d.io.read_point_cloud("D:/Data/Uni/Mobile Robotik/code/Global-Localization/test.PLY")
#o3d.visualization.draw_geometries([pcd])

abb = pcd.get_axis_aligned_bounding_box() #get bounding box
pts = np.asarray(abb.get_box_points()) #extract all 8 points
top = -0.4
bot = -0.45
print(pts)
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

cl, ind = cpcd.remove_radius_outlier(nb_points=10, radius=0.10) #remove outliers
#remove points that would not be visible anyways
#diameter = np.linalg.norm(np.asarray(cpcd.get_max_bound()) - np.asarray(cpcd.get_min_bound()))
#_, pt_map = cpcd.hidden_point_removal([0,0,diameter], diameter*100)

o3d.visualization.draw_geometries([cpcd.select_by_index(ind)]) #visualize

