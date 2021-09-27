# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2

"""
Volume Rendering
================

Example volume rendering

Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between stent-CT / brain-MRI image
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""

from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform

# # Read volume
# vol1 = np.load(io.load_data_file('volume/stent.npz'))['arr_0']
# vol2 = np.load(io.load_data_file('brain/mri.npz'))['data']
# vol2 = np.flipud(np.rollaxis(vol2, 1))

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# # Create the volume visuals, only one is visible
# atoms = np.random.rand(2, 3)
atoms = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])
molecule = scene.visuals.Molecule(atoms, parent=view.scene)

# Create three cameras (Fly, Turntable and Arcball)
fov = 00.
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                     name='Turntable')
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')
view.camera = cam2  # Select turntable at first


# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


# # create colormaps that work well for translucent and additive volume rendering
# class TransFire(BaseColormap):
    # glsl_map = """
    # vec4 translucent_fire(float t) {
        # return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    # }
    # """


# class TransGrays(BaseColormap):
    # glsl_map = """
    # vec4 translucent_grays(float t) {
        # return vec4(t, t, t, t*0.05);
    # }
    # """

# # Setup colormap iterators
# opaque_cmaps = cycle(get_colormaps())
# translucent_cmaps = cycle([TransFire(), TransGrays()])
# opaque_cmap = next(opaque_cmaps)
# translucent_cmap = next(translucent_cmaps)


# Implement axis connection with cam2
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(cam2.roll, (0, 0, 1))
        axis.transform.rotate(cam2.elevation, (1, 0, 0))
        axis.transform.rotate(cam2.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.transform.translate((50., 50.))
        axis.update()


@canvas.connect
def on_resize(event):
    molecule.shared_program["iResolution"] = event.size

if __name__ == '__main__':
    print(__doc__)
    app.run()
