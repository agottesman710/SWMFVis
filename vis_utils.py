import numpy as np
from vispy import app, scene
import os
from spacepy.pybats import bats, rim
from matplotlib import colormaps
from vispy.geometry.generation import create_sphere
from spacepy.pybats.batsmath import d_dx, d_dy


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
    return x, y, z

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
    return r, lat, lon

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


def set_shaders(mesh, diffuse=0, specular=0, ambient=1):
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
    mesh.shading_filter.ambient_light = (1, 1, 1, ambient)
    mesh.shading_filter.specular_light = (1, 1, 1, specular)

def add_earth(view):
    scene.visuals.Sphere(radius=1, method='latitude', parent=view.scene,
                                   color=[0, 0, 1, 1])

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


def create_aurora_coords(iono, views):
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
    n_aurora_coords = coords_to_xyz((90 - iono['n_theta']), iono['n_psi'], r=1.02)
    s_aurora_coords = coords_to_xyz((90 - iono['s_theta']), iono['s_psi'], r=1.02)
    n_aurora = []
    s_aurora = []
    for i in range(len(views)):
        n_aurora.append(scene.visuals.SurfacePlot(*n_aurora_coords, parent=views[i].scene))
        s_aurora.append(scene.visuals.SurfacePlot(*s_aurora_coords, parent=views[i].scene))
    return n_aurora, s_aurora


def update_aurora(iono, n_aurora, s_aurora, colormap='hot', minz=2, maxz=20):
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
    aurora_cmap = colormaps[colormap]
    n_hall_cond = iono['n_sigmah']
    n_norm_bright = (n_hall_cond - minz) / (maxz - minz)
    n_aurora_colors = aurora_cmap(n_norm_bright)
    # n_aurora_colors[:, :, 3][n_norm_bright < transparency_min] = 0

    s_hall_cond = iono['s_sigmah']
    s_norm_bright = (s_hall_cond - minz) / (maxz - minz)
    s_aurora_colors = aurora_cmap(s_norm_bright)
    # s_aurora_colors[:, :, 3][s_norm_bright < transparency_min] = 0

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