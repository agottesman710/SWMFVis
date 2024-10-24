import numpy as np
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
import vispy.io as io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import vis_utils as vutil
from vis_utils import find_facs, create_canvas, add_earth
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d
import sys

path = '/Users/arigott/Research/smc_1998/'
gm_path = path + 'GM/'

three_d_files = [f for f in sorted(os.listdir(gm_path)) if f.endswith('.out') and f.startswith('3d')]
three_d = bats.IdlFile(gm_path + three_d_files[5])

# Create a canvas with a scene
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
add_earth(view)

dist = (three_d['x'] ** 2 + three_d['y'] ** 2 + three_d['z'] ** 2) ** 0.5
ib_dist = 30
angle, dprod = find_facs(three_d)

for bool in [angle < 0.09, angle > 3.05]:
    mask = (dist > 2.5) & (dist < ib_dist) & bool

    proc_dprod = dprod[mask]

    data = np.array([three_d['x'][mask], three_d['y'][mask], three_d['z'][mask]]).T

    j = (three_d['jx'][mask] ** 2 + three_d['jy'][mask] ** 2 + three_d['jz'][mask] ** 2) ** 0.5

    norm_j = (proc_dprod - np.quantile(proc_dprod, 0.25)) / (np.quantile(proc_dprod, 0.75) - np.quantile(proc_dprod, 0.25))

    fac_mask = ((norm_j < 0.43) | (norm_j > 0.57)) & \
                ((three_d['Rho'][mask] < np.quantile(three_d['Rho'][mask], 0.8)) | (dist[mask] < 4))
    proc_data = data[fac_mask]
    proc_j = norm_j[fac_mask]

    viridis = colormaps['seismic']
    colors = viridis(proc_j)
    rgb = colors[:, :3]

    ####################
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(proc_data)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    fac_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=1)
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
    # vertices_to_remove = densities < np.quantile(densities, 0.1)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    # o3d.visualization.draw_geometries([mesh])
    fac_triangles = np.array(fac_mesh.triangles)
    fac_vertices = np.array(fac_mesh.vertices)
    mesh_colors = np.array(fac_mesh.vertex_colors)

    # Create a mesh visual
    mesh = scene.visuals.Mesh(vertices=fac_vertices, faces=fac_triangles, vertex_colors=mesh_colors, shading='smooth')
    view.add(mesh)



# Create a scatter plot
# scatter = scene.visuals.Markers()
# scatter.set_data(proc_data, edge_color=None, face_color=colors, size=5)
# view.add(scatter)



# Set the camera
view.camera = 'turntable'
view.camera.center = (0, 0, 0)
# Run the application
app.run()


