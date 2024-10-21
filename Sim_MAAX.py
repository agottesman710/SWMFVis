import numpy as np
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
import vispy.io as io
from argparse import ArgumentParser, RawDescriptionHelpFormatter

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


def create_canvas(num_cams=1, camera_altitude=7):
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                               size=(3840, 2160), show=True)
    views = []
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    for i in range(num_cams):
        views.append(scene.widgets.ViewBox(parent=canvas.scene, name=f'vb{i}',
                                           margin=0, border_color='black'))

        grid.add_widget(views[i], 0, i)
        views[i].camera = 'arcball'
        views[i].camera.interactive = False
        views[i].camera.distance = camera_altitude
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
    aurora_cmap = colormaps[colormap]
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
    n_aurora, s_aurora = create_aurora_coords(iono, views)
    update_aurora(iono, views, n_aurora, s_aurora, **kwargs)
    for i in range(len(views)):
        set_shaders(n_aurora[i])
        set_shaders(s_aurora[i])
    return n_aurora, s_aurora

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("path", type=str,
                    help='A path to an SWMF run directory.')
parser.add_argument("-o", "--outdir", default='MAAXMovieFrames/', help="Set " +
                    "output file name.  Defaults to 'MAAXMovieFrames/'")
parser.add_argument("-minz", "--minz", default=2, type=int,
                    help="Sets the minimum value for the aurora colormap. " +
                    "Default is 2.")
parser.add_argument("-maxz", "--maxz", default=20, type=int,
                    help="Sets the maximum value for the aurora colormap. " +
                    "Default is 20.")
parser.add_argument("-c", "--cameras", default=2, type=int,
                    help="Sets the number of cameras to view Earth from. " +
                    "Default is 2 cameras for the MAAX configuration.")
parser.add_argument("-cr", "--camera_rotation", default=180, type=float,
                    help="Sets the camera rotation from one camera to the next. " +
                    "Set this argument to a float degree value to change. Negative " +
                    "values are recommended so general flow is left to right. " +
                    "Default is -180 degrees, or opposite sides of the Earth.")
parser.add_argument("-ca", "--camera_altitude", default=7, type=float,
                    help="Sets the camera altitude from the Earth. Default is 7.")
parser.add_argument("-start", "--starting_index", default=0, type=int,
                    help="Sets the starting index of iono files - Mainly useful when " +
                         "restarting the run. Default is 0.")
args = parser.parse_args()

path = args.path
if path[-1] != '/':
    path += '/'
ie_path = path + 'IE/'
outdir = args.outdir

iono_files = [f for f in sorted(os.listdir(ie_path)) if f.endswith('.idl.gz')]
iono_start = rim.Iono(ie_path + iono_files[0])

aurora_canvas, sat_views = create_canvas(args.cameras, camera_altitude=args.camera_altitude)

n_auroras, s_auroras = create_aurora(iono_start, sat_views, minz=args.minz, maxz=args.maxz)

for j, view in enumerate(sat_views):
    view.camera.transform.rotate(args.camera_rotation * j, [1, 0, 0])
    aurora_canvas.render()

if not os.path.exists(outdir):
    os.makedirs(outdir)

for i in range(args.starting_index, len(iono_files)):

    for view in sat_views:
        view.camera.transform.rotate(0.23, [1, 0, 0])
    iono = rim.Iono(ie_path + iono_files[i])
    update_aurora(iono, sat_views, n_auroras, s_auroras, maxz=40)
    aurora_canvas.update()
    img = aurora_canvas.render()

    io.image.imsave(f'{outdir}/frame{i:04d}.png', img)
