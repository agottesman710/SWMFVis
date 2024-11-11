import numpy as np
from matplotlib.pyplot import pause
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
import vispy.io as io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import vis_utils as vutil
from vis_utils import find_facs, create_canvas, add_earth, find_outer_points, create_aurora
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d
import sys



parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("path", type=str,
                    help='A path to an SWMF run directory.')

args = parser.parse_args()
path = args.path
if path[-1] != '/':
    path += '/'
gm_path = path + 'GM/'
time_step = 5

three_d_files = [f for f in sorted(os.listdir(gm_path)) if f.endswith('.out') and f.startswith('3d')]
three_d = bats.IdlFile(gm_path + three_d_files[time_step])
iono_filename = ('it' + three_d_files[time_step].split('_e')[1][2:8] + '_' +
                 three_d_files[time_step].split('-')[1] + "_000.idl.gz")


#%% Establish some variables
dist = (three_d['x'] ** 2 + three_d['y'] ** 2 + three_d['z'] ** 2) ** 0.5

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

angle, dprod = find_facs(three_d)

#%% FACs
ib_dist = 30


meshes = []
for bool in [angle < 0.09, angle > 3.05]:
    mask = (dist > 2.5) & (dist < ib_dist) & bool

    data = np.array([three_d['x'][mask], three_d['y'][mask], three_d['z'][mask]]).T

    j = (three_d['jx'][mask] ** 2 + three_d['jy'][mask] ** 2 + three_d['jz'][mask] ** 2) ** 0.5

    norm_no_mask = (dprod - np.quantile(dprod, 0.15)) / (np.quantile(dprod, 0.85) - np.quantile(dprod, 0.15))
    norm_j = norm_no_mask[mask]

    fac_mask = ((norm_j < 0.49) | (norm_j > 0.51)) & \
                ((three_d['Rho'][mask] < np.quantile(three_d['Rho'][mask], 0.8)) | (dist[mask] < 4))
    proc_data = data[fac_mask]
    proc_j = norm_j[fac_mask]

    diverging = colormaps['seismic']
    colors = diverging(proc_j)
    rgb = colors[:, :3]

    ####################
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(proc_data)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    fac_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=1)
    triangle_clusters, cluster_n_triangles, cluster_area = (fac_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    fac_mesh.remove_triangles_by_mask(triangles_to_remove)

    fac_triangles = np.array(fac_mesh.triangles)
    fac_vertices = np.array(fac_mesh.vertices)
    mesh_colors = np.array(fac_mesh.vertex_colors)
    mask = (fac_vertices[fac_triangles[:, 0]][:, 0] ** 2 + fac_vertices[fac_triangles[:, 0]][:, 1] ** 2 +
            fac_vertices[fac_triangles[:, 0]][:, 2] ** 2 > 4 ** 2) & \
           (fac_vertices[fac_triangles[:, 1]][:, 0] ** 2 + fac_vertices[fac_triangles[:, 1]][:, 1] ** 2 +
            fac_vertices[fac_triangles[:, 1]][:, 2] ** 2 > 4 ** 2) & \
           (fac_vertices[fac_triangles[:, 2]][:, 0] ** 2 + fac_vertices[fac_triangles[:, 2]][:, 1] ** 2 +
            fac_vertices[fac_triangles[:, 2]][:, 2] ** 2 > 4 ** 2)
    fac_triangles = fac_triangles[mask]
    # Create a mesh visual
    mesh = scene.visuals.Mesh(vertices=fac_vertices, faces=fac_triangles, vertex_colors=mesh_colors, shading='smooth')
    meshes.append(mesh)


def update_fac1():
    if meshes[0].visible:
        meshes[0].visible = False
    else:
        meshes[0].visible = True

def update_fac2():
    if meshes[1].visible:
        meshes[1].visible = False
    else:
        meshes[1].visible = True

#%% Chapman-Ferraro


mask = dist > 2.5
j = (three_d['jx'][mask] ** 2 + three_d['jy'][mask] ** 2 + three_d['jz'][mask] ** 2) ** 0.5
density = three_d['Rho'][mask]
ux = three_d['Ux'][mask]

data = np.array([three_d['x'][mask], three_d['y'][mask], three_d['z'][mask]]).T


# norm_pressure = (pressure - np.quantile(pressure, 0.25)) / (np.quantile(pressure, 0.75) - np.quantile(pressure, 0.25))
# norm_ux = (ux - np.quantile(ux, 0.25)) / (np.quantile(ux, 0.75) - np.quantile(ux, 0.25))

diverging = colormaps['seismic']
blues = colormaps['Blues']


####################
# BRENNER ET AL 2021 - FRONTIERS
####################
thermal_pressure = three_d['P'][mask]
u = (three_d['Ux'][mask] ** 2 + three_d['Uy'][mask] ** 2 + three_d['Uz'][mask] ** 2) ** 0.5
plasma_pressure = three_d['Rho'][mask] * 1e6 * 1.67e-27 * ((u * 1e3) ** 2)
b = (three_d['Bx'][mask] ** 2 + three_d['By'][mask] ** 2 + three_d['Bz'][mask] ** 2) ** 0.5
plasma_beta = (thermal_pressure * 1e-9 + plasma_pressure) / ((b * 1e-9) ** 2 / (2 * 4 * np.pi * 1e-7))

ti = three_d['P'][mask] * 1e-9 / (three_d['Rho'][mask] * 1e6 * 1.6e-19)

pause_mask = (plasma_beta > 0.65) & (plasma_beta < 0.75)

pause_data = data[pause_mask]
pause_current = j[pause_mask]
pause_density = density[pause_mask]
outer_mask = find_outer_points(pause_data[:, 0], pause_data[:, 1], pause_data[:, 2])
pause_data = pause_data[outer_mask]
pause_current = pause_current[outer_mask]
pause_density = pause_density[outer_mask]
norm_pause_current = (pause_current - np.quantile(pause_current, 0.15)) / (np.quantile(pause_current, 0.85) - np.quantile(pause_current, 0.15))
norm_pause_density = (pause_density - np.quantile(pause_density, 0)) / (np.quantile(pause_density, 0.5) - np.quantile(pause_density, 0))
colors = blues(norm_pause_density)
rgb = colors[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pause_data)
pcd.colors = o3d.utility.Vector3dVector(rgb)


cf_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=20)

cf_triangles = np.array(cf_mesh.triangles)
cf_vertices = np.array(cf_mesh.vertices)
norm_x = (cf_vertices[:, 0] - cf_vertices[:, 0].min()) / (cf_vertices[:, 0].max() * 3 - cf_vertices[:, 0].min())
mesh_colors = np.array(cf_mesh.vertex_colors)
new_colors = np.zeros((mesh_colors.shape[0], 4))
new_colors[:, :3] = mesh_colors[:, :3]
new_colors[:, 3] = norm_x - 0.1

# Create a mesh visual
cf_mesh = scene.visuals.Mesh(vertices=cf_vertices, faces=cf_triangles, vertex_colors=new_colors, shading='smooth')
cf_mesh.shading_filter.diffuse_light = (1, 1, 1, 1)
cf_mesh.shading_filter.light_dir = (-10, 0, 0)
cf_mesh.shading_filter.ambient_light = (1, 1, 1, 0)
cf_mesh.shading_filter.specular_light = (1, 1, 1, 0)

def update_cf():
    if cf_mesh.visible:
        cf_mesh.visible = False
    else:
        cf_mesh.visible = True

#%% Plasma sheet

jy = three_d['jy'][mask] / j
proc_angle = angle[mask]
sheet_mask = (ux > 200) & (jy > 0.95) # & (plasma_beta > 5)
sheet_data = data[sheet_mask]
sheet_current = j[sheet_mask]
sheet_density = density[sheet_mask]
sheet_ux = ux[sheet_mask]

norm_ux = sheet_ux / (np.quantile(sheet_ux, 0.85))
colors = blues(norm_ux)
sheet_rgb = colors[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sheet_data)
pcd.colors = o3d.utility.Vector3dVector(sheet_rgb)
pcd.estimate_normals()
sheet_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=1)
sheet_mesh = sheet_mesh.filter_smooth_taubin(number_of_iterations=20)
sheet_triangles = np.array(sheet_mesh.triangles)
sheet_vertices = np.array(sheet_mesh.vertices)
mesh_colors = np.array(sheet_mesh.vertex_colors)
sheet_mesh = scene.visuals.Mesh(vertices=sheet_vertices, faces=sheet_triangles, vertex_colors=mesh_colors, shading='smooth')

def update_plasma_sheet():
    if sheet_mesh.visible:
        sheet_mesh.visible = False
    else:
        sheet_mesh.visible = True

#%% Put Everything together

# Create a canvas with a scene
canvas = scene.SceneCanvas(keys={'f': update_fac1, 'g': update_fac2, 'c': update_cf, 'p': update_plasma_sheet},
                           show=True)
view = canvas.central_widget.add_view()

widget = scene.Widget(pos=(0, 0), size=(200, 200), bgcolor='k', parent=canvas.scene)
# sub view
view2 = widget.add_view()
view2.camera = 'turntable'
axis2 = scene.visuals.XYZAxis(width=2, parent=view2.scene)


# add_earth(view)

# Create aurora
iono = rim.Iono(path + 'IE/' + iono_filename)
n_auroras, s_auroras = create_aurora(iono, [view], minz=5, maxz=25, colormap2='hot')



view.add(sheet_mesh)

view.add(meshes[0])
view.add(meshes[1])

view.add(cf_mesh)


# Create a scatter plot
# scatter = scene.visuals.Markers()
# scatter.set_data(proc_data, edge_color=None, face_color=colors, size=5)
# view.add(scatter)



# Set the camera
view.camera = 'turntable'
view.camera.center = (0, 0, 0)
view.camera.link(view2.camera)

# Run the application
app.run()


