import numpy as np
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
import vispy.io as io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import vis_utils as vutil
from vis_utils import coords_to_xyz, set_shaders, create_canvas, create_aurora_coords, update_aurora, create_aurora, add_earth
from vispy.geometry import create_sphere

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("path", type=str,
                    help='A path to an SWMF run directory.')
parser.add_argument("-o", "--outdir", default='MAAXMovieFrames/', help="Set " +
                    "output file name.  Defaults to 'MAAXMovieFrames/'")
parser.add_argument("-minz", "--minz", default=2, type=float,
                    help="Sets the minimum value for the aurora colormap. " +
                    "Default is 2.")
parser.add_argument("-maxz", "--maxz", default=20, type=float,
                    help="Sets the maximum value for the aurora colormap. " +
                    "Default is 20.")
parser.add_argument("-c", "--cameras", default=2, type=int,
                    help="Sets the number of cameras to view Earth from. " +
                    "Default is 2 cameras for the MAAX configuration.")
parser.add_argument("-cs", "--camera_separation", default=-180, type=float,
                    help="Sets the camera angular separation from one camera to the next. " +
                    "Set this argument to a float degree value to change. Negative " +
                    "values are recommended so general flow is left to right. " +
                    "Default is -180 degrees, or opposite sides of the Earth.")
parser.add_argument("-cr", "--camera_rotation", default=None, type=float,
                    help="Sets the rotation speed of the cameras per time step. " +
                    "Set this argument to a float degree value to set it as a constant" +
                    "otherwise it will calculate the rotation based on the camera altitude. " +
                    "Default is calculating the rotation based on the camera altitude.")
parser.add_argument("-ca", "--camera_altitude", default=7, type=float,
                    help="Sets the camera altitude from the Earth. Default is 7.")
parser.add_argument("-start", "--starting_index", default=0, type=int,
                    help="Sets the starting index of iono files - Mainly useful when " +
                         "restarting the run. Default is 0.")
parser.add_argument("-cb", "--colorbar", default=False, type=bool,
                    help="Adds a color bar. Default is false.")
args = parser.parse_args()

path = args.path
if path[-1] != '/':
    path += '/'
ie_path = path + 'IE/'
outdir = args.outdir

iono_files = [f for f in sorted(os.listdir(ie_path)) if f.endswith('.idl.gz')]
iono_start = rim.Iono(ie_path + iono_files[0])

aurora_canvas, sat_views = create_canvas(args.cameras, camera_altitude=args.camera_altitude)

for j, view in enumerate(sat_views):
    view.camera.transform.rotate(args.camera_separation * j, [1, 0, 0])
    aurora_canvas.render()

if not os.path.exists(outdir):
    os.makedirs(outdir)

rotation = None
if args.camera_rotation is None:
    gm = 3.986e14
    r_earth = 6371000
    v = np.sqrt(gm / (r_earth * args.camera_altitude))
    rotation = v / (r_earth * args.camera_altitude) * 180 / np.pi * 60
else:
    rotation = args.camera_rotation

for view in sat_views:
    view.camera.transform.rotate(rotation * args.starting_index, [1, 0, 0])
    # add_earth(view)

n_auroras, s_auroras = create_aurora(iono_start, sat_views, minz=args.minz, maxz=args.maxz)

for i in range(args.starting_index, len(iono_files)):
    for view in sat_views:
        view.camera.transform.rotate(rotation, [1, 0, 0])
    iono = rim.Iono(ie_path + iono_files[i])
    update_aurora(iono, n_auroras, s_auroras, maxz=args.maxz, minz=args.minz)
    aurora_canvas.update()
    img = aurora_canvas.render()

    io.image.imsave(f'{outdir}/frame{i:04d}.png', img)
