"""
Create a PseudoSpheres Visual
"""
import numpy as np
from vispy import app, scene

# Create canvas and view
canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.ArcballCamera(fov=0)
view.camera.scale_factor = 70

# Prepare data
np.random.seed(777)
data = np.random.normal(size=(40, 3), loc=0, scale=10)
size = np.random.rand(40) * 10
colors = np.random.rand(40, 3)

# Create and show visual
vis = scene.visuals.PseudoSpheres(pos=data, size=size, color=colors)
vis.parent = view.scene

lines = np.array([[data[i], data[-1]]
                  for i in range(len(data) - 1)])
line_vis = []
for line in lines:
    vis2 = scene.visuals.Tube(line, radius=.4)
    vis2.parent = view.scene
    line_vis.append(vis2)


# Run example
app.run()
