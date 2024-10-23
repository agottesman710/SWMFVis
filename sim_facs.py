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

path = '/Users/arigott/PycharmProjects/SpacepyAurora/run_noBY/'
gm_path = path + 'GM/'

three_d_files = [f for f in sorted(os.listdir(gm_path)) if f.endswith('.out') and f.startswith('3d')]
three_d = bats.IdlFile(gm_path + three_d_files[5])

dist = (three_d['x'] ** 2 + three_d['y'] ** 2 + three_d['z'] ** 2) ** 0.5
ib_dist = 30
angle, dprod = find_facs(three_d)
mask = (dist > 2.5) & (dist < ib_dist) & ((angle < 0.09) | (angle > 3.05))

proc_dprod = dprod[mask][::10]

data = np.array([three_d['x'][mask][::10], three_d['y'][mask][::10], three_d['z'][mask][::10]]).T

j = (three_d['jx'][mask][::10] ** 2 + three_d['jy'][mask][::10] ** 2 + three_d['jz'][mask][::10] ** 2) ** 0.5

norm_j = (proc_dprod - np.quantile(proc_dprod, 0.25)) / (np.quantile(proc_dprod, 0.75) - np.quantile(proc_dprod, 0.25))

fac_mask = ((norm_j < 0.43) | (norm_j > 0.57)) & \
            ((three_d['Rho'][mask][::10] < np.quantile(three_d['Rho'][mask][::10], 0.8)) | (dist[mask][::10] < 4))
proc_data = data[fac_mask]
proc_j = norm_j[fac_mask]

viridis = colormaps['seismic']
colors = viridis(proc_j)

# Create a canvas with a scene
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
add_earth(view)

# Create a scatter plot
scatter = scene.visuals.Markers()
scatter.set_data(proc_data, edge_color=None, face_color=colors, size=5)
view.add(scatter)


# Set the camera
view.camera = 'turntable'
view.camera.center = (0, 0, 0)
# Run the application
app.run()


