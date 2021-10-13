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


from ..gloo import RenderBuffer, FrameBuffer, Texture2D
from ..visuals.shaders import Variable, ModularProgram


vert_ssao = """#version 450
attribute vec2 a_position;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(a_position, 0, 1);
    v_texcoord = (a_position + 1) / 2;
}
"""

frag_ssao = """#version 450
uniform sampler2D tex_color;
uniform sampler2D tex_normal_depth;

in vec2 v_texcoord;

void main() {
    vec4 c = texture2D(tex_color, v_texcoord);
    //vec4 d = texture2D(tex_normal_depth, v_texcoord);
    $out_color = vec4(c);
    //gl_FragDepth = -1;
}
"""


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

        width, height = canvas.size

        # A quad (two triangles) spanning the framebuffer area.
        self.screen_vertices = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        # ambient_occlusion
        self.ssao_prog = ModularProgram(vcode=vert_ssao, fcode=frag_ssao)
        self.ssao_prog['a_position'] = self.screen_vertices
        self.color_texture = Texture2D((height, width, 4), format='rgba')
        self.ssao_prog['tex_color'] = self.color_texture
        self.normal_depth_texture = Texture2D((height, width, 4), format='rgba')
        self.ssao_prog['tex_normal_depth'] = self.normal_depth_texture

        self.out_color = Variable('out vec4 out_color')
        self.ssao_prog.frag['out_color'] = self.out_color

        self.depth_buffer = RenderBuffer((height, width), 'depth')
        self.framebuffer = FrameBuffer(
            color=self.color_texture, #self.normal_depth_texture],
            depth=self.depth_buffer
        )

    def resize(self, size):
        height_width = size[::-1]
        channels = self.color_texture.shape[2:]
        self.depth_buffer.resize(height_width)
        self.color_texture.resize(height_width + channels)
        self.normal_depth_texture.resize(height_width + channels)

    def render(self, bgcolor):
        canvas = self.canvas

        offset = 0, 0
        canvas_size = canvas.size

        def push_fbo():
            canvas.push_fbo(self.framebuffer, offset, canvas_size)

        def pop_fbo():
            canvas.pop_fbo()

        # draw to textures
        canvas.context.set_state(
            depth_mask=True,
        )
        push_fbo()
        canvas.context.clear(color=bgcolor)
        canvas.draw_visual(canvas.scene)
        pop_fbo()

        # # draw on canvas back buffer
        canvas.context.set_state(
            depth_test=True,
            blend=True,
            depth_mask=True,
        )
        canvas.context.clear(color=bgcolor)
        self.ssao_prog.draw('triangle_strip')
