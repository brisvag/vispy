# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""A renderer supporting weighted blended order-independent transparency.

The computed transparency is approximate but emulates transparency well in most
cases.

One artefact that might arise is that a transparent object very close to the
camera might appear opaque and occlude the background. The solution is to
ensure some distance between the camera and the object.

Notes
-----
.. [1] McGuire, Morgan, and Louis Bavoil. "Weighted blended order-independent
   transparency." Journal of Computer Graphics Techniques 2.4 (2013).
"""

import numpy as np

from vispy import gloo


vert_ssao = """
attribute vec2 a_position;

varying vec2 v_texcoord;

void main() {
    gl_Position = vec4(a_position, 0, 1);
    v_texcoord = (a_position + 1) / 2;
}
"""

frag_ssao = """
uniform sampler2D tex_depth;
varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(tex_depth, v_texcoord) + 0.5;
}
"""


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

        # TODO: Observe when a new scene is set in the canvas, and update the
        # callback.
        canvas.scene.events.children_change.connect(self.on_scene_changed)

        width, height = canvas.size

        self.depth_buffer = gloo.Texture2D((height, width), format='alpha')
        self.color_buffer = gloo.Texture2D((height, width, 4), format='rgba')
        self.framebuffer = gloo.FrameBuffer(
            color=self.color_buffer,
            depth=self.depth_buffer,
        )

        # A quad (two triangles) spanning the framebuffer area.
        self.screen_vertices = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        # ambient_occlusion
        self.normal_buffer = gloo.Texture2D((*self.canvas.size[::-1], 4))

        self.ssao_prog = gloo.Program(vert_ssao, frag_ssao)
        self.ssao_prog['a_position'] = self.screen_vertices
        self.ssao_prog['tex_depth'] = self.depth_buffer

        self._scene_changed = True

    def on_scene_changed(self, event):
        self._scene_changed = True

    def resize(self, size):
        height_width = size[::-1]
        channels = self.color_buffer.shape[2:]
        self.color_buffer.resize(height_width + channels)
        self.depth_buffer.resize(height_width)
        channels = self.normal_buffer.shape[2:]
        self.normal_buffer.resize(height_width + channels)

    def render(self, bgcolor):
        canvas = self.canvas

        offset = 0, 0
        canvas_size = self.canvas.size

        def push_fbo():
            canvas.push_fbo(self.framebuffer, offset, canvas_size)

        def pop_fbo():
            canvas.pop_fbo()

        def iter_node_tree(node):
            if hasattr(node, '_get_hook'):
                yield node
            for child in node.children:
                yield from iter_node_tree(child)

        # for visual in iter_node_tree(canvas.scene):
            # hook = visual._get_hook('frag', 'post')

        # push_fbo()
        canvas.context.clear(color=bgcolor, depth=True)
        self.canvas.draw_visual(canvas.scene)
        # pop_fbo()

        # push_fbo()
        self.ssao_prog.draw('triangle_strip')
        # pop_fbo()
