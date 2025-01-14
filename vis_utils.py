import numpy as np
from vispy import app, scene
from vispy.io import imread
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
from vispy.geometry.generation import create_sphere
from spacepy.pybats.batsmath import d_dx, d_dy
from vispy.visuals.filters import TextureFilter
from spacepy.coordinates import Coords
from spacepy.time import Ticktock


#%% Point Stuffs

def coords_to_xyz(lat, lon, r=1.0):
    """
    Convert latitude and longitude to cartesian coordinates.

    Parameters
    ----------
    lat - latitude on sphere
    lon - longitude on sphere
    r - radius of the sphere - default is 1.0 for plotting Earth

    Returns
    -----------
    x, y, z - cartesian coordinates
    """
    lat *= np.pi / 180
    lon *= np.pi / 180
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])

def cartesian_to_spherical(x, y, z):
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z - cartesian coordinates

    Returns
    -----------
    lat, lon - latitude and longitude
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return np.array([r, lat, lon])

def find_outer_points(x, y, z, num_bins=45):
    r, theta, phi = cartesian_to_spherical(x, y, z)
    min_phi = phi[r[x < 0].argmax()]
    theta_bins = np.linspace(-90, 90, num_bins)
    phi_bins = np.linspace(-np.abs(min_phi), np.abs(min_phi), num_bins)
    outer_mask = np.zeros_like(x, dtype=bool)
    for i in range(num_bins - 1):
        for j in range(num_bins - 1):
            steradian_mask = ((theta >= theta_bins[i]) & (theta < theta_bins[i + 1]) &
                              (phi >= phi_bins[j]) & (phi < phi_bins[j + 1]))

            if np.any(steradian_mask):
                max_index = np.argmax(r[steradian_mask])
                outer_mask[np.where(steradian_mask)[0][max_index]] = True
    return outer_mask


#%% Vispy Stuffs


def set_shaders(mesh, diffuse=0.0, specular=0.0, ambient=1.0, light_dir=(-10, 0, 0)):
    """
    Sets the shading of a mesh object based on parameters.

    Parameters
    ----------
    mesh - the object to change the shading of
    diffuse - the diffuse light coefficient, changes the brightness of the directional light on the object
    - default is 0
    specular - the specular light coefficient, changes the brightness of the reflective light on the object
    - default is 0
    ambient - changes the consistent light on the object - default is 1

    Defaults create a consistent lighting with no shading - looks 2D

    Returns
    -----------
    None
    ""
    """
    mesh.shading_filter.diffuse_light = (1, 1, 1, diffuse)
    mesh.shading_filter.light_dir = light_dir
    mesh.shading_filter.ambient_light = (1, 1, 1, ambient)
    mesh.shading_filter.specular_light = (1, 1, 1, specular)

def add_earth(view, texture_path=None):
    earth = scene.visuals.Sphere(radius=1, method='latitude', parent=view.scene,
                                   color=[1, 1, 1, 1], shading='smooth', cols=2048, rows=1024)
    if texture_path is not None:
        texture = imread(texture_path)
        longitude = np.linspace(0, 1, texture.shape[0])
        latitude = np.linspace(0, 1, texture.shape[1])
        texcoords = np.dstack(np.meshgrid(latitude, longitude)).reshape(-1, 2)
        texcoords = np.delete(texcoords, np.arange(4096, texcoords.shape[0], 4096), axis=0)
        texture_filter = TextureFilter(texture, texcoords=texcoords)
        earth.mesh.attach(texture_filter)
    set_shaders(earth.mesh, diffuse=1, ambient=0.2)
    return earth

def create_canvas(num_cams=1, camera_altitude=7, camera_type='arcball', resolution=(3840, 2160), interactive=False,
                  center=(0, 0, 0), fov=20):
    """
    Create a canvas with multiple views as children of the canvas

    Parameters
    --------------
    num_cams - the number of cameras to view the Earth from - default is 1
    camera_altitude - the altitude of the camera from the Earth - default is 7
    camera_type - the type of camera to use - default is 'arcball'

    Returns
    --------------
    canvas, views - the canvas and the array of view objects for the cameras
    """
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                               size=resolution, show=True)
    views = []
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    for i in range(num_cams):
        views.append(scene.widgets.ViewBox(parent=canvas.scene, name=f'vb{i}',
                                           margin=0, border_color='black'))

        grid.add_widget(views[i], 0, i)
        views[i].camera = camera_type
        views[i].camera.interactive = interactive
        views[i].camera.distance = camera_altitude
        views[i].camera.center = center
        views[i].camera.fov = fov
    return canvas, views


def create_aurora_coords(iono, views, GEO=True):
    """
    Create the aurora coordinates for the aurora surface plots from iono coordinate data

    Parameters
    --------------
    iono - the iono object with the aurora data
    views - the views to add the aurora coordinates to

    Returns
    --------------
    n_aurora, s_aurora - arrays of the coordinates for the north and south aurora surface
    """
    if GEO:
        n_gsm_coords = np.array([np.full((iono['n_theta'].shape[0] * iono['n_theta'].shape[1]), 1.02),
                                 90 - iono['n_theta'].flatten(), 180 - iono['n_psi'].flatten()]).T
        n_gsm_coords = Coords(n_gsm_coords, 'SM', 'sph', ['Re', 'deg', 'deg'])
        n_gsm_coords.ticks = Ticktock([iono.attrs['time']] * n_gsm_coords.long.shape[0], 'UTC')
        n_geo_coords = n_gsm_coords.convert('GEO', 'sph')
        n_aurora_coords = coords_to_xyz(n_geo_coords.lati, n_geo_coords.long, r=1.02)
        n_aurora_coords = n_aurora_coords.reshape(3, iono['n_theta'].shape[0], iono['n_theta'].shape[1])

        s_gsm_coords = np.array([np.full((iono['s_theta'].shape[0] * iono['s_theta'].shape[1]), 1.02),
                                 90 - iono['s_theta'].flatten(), 180 - iono['s_psi'].flatten()]).T
        s_gsm_coords = Coords(s_gsm_coords, 'SM', 'sph', ['Re', 'deg', 'deg'])
        s_gsm_coords.ticks = Ticktock([iono.attrs['time']] * s_gsm_coords.long.shape[0], 'UTC')
        s_geo_coords = s_gsm_coords.convert('GEO', 'sph')
        s_aurora_coords = coords_to_xyz(s_geo_coords.lati, s_geo_coords.long, r=1.02)
        s_aurora_coords = s_aurora_coords.reshape(3, iono['s_theta'].shape[0], iono['s_theta'].shape[1])

        # print(n_aurora_coords[0].max(), n_aurora_coords[0].min())
        # print(n_aurora_coords[1].max(), n_aurora_coords[1].min())
        # print(n_aurora_coords[2].max(), n_aurora_coords[2].min())
        # print(s_aurora_coords[0].max(), s_aurora_coords[0].min())
        # print(s_aurora_coords[1].max(), s_aurora_coords[1].min())
        # print(s_aurora_coords[2].max(), s_aurora_coords[2].min())
    else:
        n_aurora_coords = coords_to_xyz((90 - iono['n_theta']), iono['n_psi'], r=1.02)
        s_aurora_coords = coords_to_xyz((90 - iono['s_theta']), iono['s_psi'], r=1.02)
    n_aurora = []
    s_aurora = []
    for i in range(len(views)):
        n_aurora.append(scene.visuals.SurfacePlot(*n_aurora_coords, parent=views[i].scene))
        s_aurora.append(scene.visuals.SurfacePlot(*s_aurora_coords, parent=views[i].scene))
    return n_aurora, s_aurora


def update_aurora(iono, n_aurora, s_aurora, colormap1='hot', colormap2='hot', minz=2, maxz=20, transparency_min=0.0):
    """
    Update the aurora surface plots with new iono conductivity data

    Parameters
    --------------
    iono - the iono object with the aurora conductivity data
    n_aurora - the array of north aurora surface plot objects
    s_aurora - the array of  south aurora surface plot objects
    colormap - the colormap to use for the aurora - default is 'viridis'
    minz - the minimum value for the colormap - default is 2
    maxz - the maximum value for the colormap - default is 20

    Returns
    --------------
    None
    """
    # iono.calc_bright()
    aurora_cmap = colormaps[colormap1]
    n_hall_cond = iono['n_sigmah']
    n_norm_bright = (n_hall_cond - minz) / (maxz - minz)
    n_aurora_colors = aurora_cmap(n_norm_bright)
    n_aurora_colors[:, :, 3] = n_norm_bright + 0.2
    n_aurora_colors[:, :, 3][n_norm_bright < transparency_min] = 0

    aurora_cmap = colormaps[colormap2]
    s_hall_cond = iono['s_sigmah']
    s_norm_bright = (s_hall_cond - minz) / (maxz - minz)
    s_aurora_colors = aurora_cmap(s_norm_bright)
    s_aurora_colors[:, :, 3] = s_norm_bright + 0.2
    s_aurora_colors[:, :, 3][s_norm_bright < transparency_min] = 0

    for i in range(len(n_aurora)):
        n_aurora[i].set_data(colors=n_aurora_colors)
        s_aurora[i].set_data(colors=s_aurora_colors)


def create_aurora(iono, views, **kwargs):
    n_aurora, s_aurora = create_aurora_coords(iono, views)
    update_aurora(iono, n_aurora, s_aurora, **kwargs)
    for i in range(len(views)):
        views[i].add(n_aurora[i])
        views[i].add(s_aurora[i])
        set_shaders(n_aurora[i])
        set_shaders(s_aurora[i])
    return n_aurora, s_aurora


#%% Spacepy Stuffs
def find_facs(three_d):
    """
    Finds field aligned currents by using dot product to find angle between current and B-field

    Parameters
    --------------
    three_d - the pybats object of a 3D data file

    Returns
    --------------
    facs - the field aligned currents
    """

    jx, jy, jz = three_d['jx'], three_d['jy'], three_d['jz']
    bx, by, bz = three_d['Bx'], three_d['By'], three_d['Bz']
    jmag = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    jmag[jmag < 1e-12] = np.nan
    bmag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    bhx, bhy, bhz = bx / bmag, by / bmag, bz / bmag

    dprod = jx * bhx + jy * bhy + jz * bhz
    angle = np.arccos((dprod / jmag))

    return angle, dprod