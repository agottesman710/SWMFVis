import numpy as np
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
import matplotlib.cm as cm
import imageio
from vispy.gloo.util import _screenshot
import vispy.io as io


def coords_to_xyz(lat, lon, r=1.0):
    lat *= np.pi / 180
    lon *= np.pi / 180
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def set_shaders(mesh):
    mesh.shading_filter.diffuse_light = (1, 1, 1, 0)
    mesh.shading_filter.ambient_light = (1, 1, 1, 1)
    mesh.shading_filter.specular_light = (1, 1, 1, 0)


def create_canvas(num_cams=1):
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                               size=(3840, 2160), show=True)
    views = []
    for i in range(num_cams):
        w, h = canvas.size
        w2 = w / 2
        h2 = h / 2
        views.append(scene.widgets.ViewBox(parent=canvas.scene, name=f'vb{i}',
                                           margin=0, border_color='black'))
        views[i].pos = w2 * i, 0
        views[i].size = w2, h
        views[i].camera = 'arcball'
        views[i].camera.interactive = False
        views[i].camera.distance = 7
        views[i].camera.center = (0, 0, 0)
    return canvas, views


def create_aurora_coords(iono, views):
    n_aurora_coords = coords_to_xyz((90 - iono['n_theta']), iono['n_psi'], r=1.01)
    s_aurora_coords = coords_to_xyz((90 - iono['s_theta']), iono['s_psi'], r=1.01)
    n_aurora = []
    s_aurora = []
    for i in range(len(views)):
        n_aurora.append(scene.visuals.SurfacePlot(*n_aurora_coords, parent=views[i].scene))
        s_aurora.append(scene.visuals.SurfacePlot(*s_aurora_coords, parent=views[i].scene))
    return n_aurora, s_aurora


def update_aurora(iono, views, n_aurora, s_aurora, colormap='viridis', minz=2, maxz=20):
    aurora_cmap = cm.get_cmap(colormap)
    n_hall_cond = iono['n_sigmah']
    n_norm_bright = (n_hall_cond - minz) / (maxz - minz)
    n_aurora_colors = aurora_cmap(n_norm_bright)

    s_hall_cond = iono['s_sigmah']
    s_norm_bright = (s_hall_cond - minz) / (maxz - minz)
    s_aurora_colors = aurora_cmap(s_norm_bright)

    for i in range(len(views)):
        n_aurora[i].set_data(colors=n_aurora_colors[:, :, :3])
        views[i].add(n_aurora[i])
        s_aurora[i].set_data(colors=s_aurora_colors[:, :, :3])
        views[i].add(s_aurora[i])


def create_aurora(iono, views, **kwargs):
    n_aurora, s_aurora = create_aurora_coords(iono_naught, views)
    update_aurora(iono, views, n_aurora, s_aurora, **kwargs)
    for i in range(len(views)):
        set_shaders(n_aurora[i])
        set_shaders(s_aurora[i])
    return n_aurora, s_aurora

path = '/Users/ari/PycharmProjects/SWMF_Aurora/Gannon Run/'
gm_path = path + 'GM/'
ie_path = path + 'IE/'

iono_files = [f for f in sorted(os.listdir(ie_path)) if f.endswith('.idl')]
iono_naught = rim.Iono(ie_path + iono_files[0])

canvas, views = create_canvas(2)
views[1].camera.transform.rotate(90, [1, 0, 0])

n_auroras, s_auroras = create_aurora(iono_naught, views, maxz=40)

for i in range(len(iono_files)):
    for view in views:
        view.camera.transform.rotate(0.23, [1, 0, 0])
    iono = rim.Iono(ie_path + iono_files[i])
    update_aurora(iono, views, n_auroras, s_auroras, maxz=40)
    canvas.update()
    img = canvas.render()
    io.image.imsave(f'/Users/ari/Documents/GannonMovie90/frame{i:04d}.png', img)
