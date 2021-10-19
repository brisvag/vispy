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
uniform sampler2D tex_noise;
uniform sampler2D tex_color;
uniform sampler2D tex_normal_depth;

uniform float base;
uniform float radius;
uniform float falloff;
uniform float strength;

in vec2 v_texcoord;

void main(void) {
  // Samples count
  const int samples = 16;
  // Random vectors inside a unit sphere
  const vec3 sample_kernel[samples] = vec3[](
      vec3( 0.5381, 0.1856,-0.4319), vec3( 0.1379, 0.2486, 0.4430),
      vec3( 0.3371, 0.5679,-0.0057), vec3(-0.6999,-0.0451,-0.0019),
      vec3( 0.0689,-0.1598,-0.8547), vec3( 0.0560, 0.0069,-0.1843),
      vec3(-0.0146, 0.1402, 0.0762), vec3( 0.0100,-0.1924,-0.0344),
      vec3(-0.3577,-0.5301,-0.4358), vec3(-0.3169, 0.1063, 0.0158),
      vec3( 0.0103,-0.5869, 0.0046), vec3(-0.0897,-0.4940, 0.3287),
      vec3( 0.7119,-0.0154,-0.0918), vec3(-0.0533, 0.0596,-0.5411),
      vec3( 0.0352,-0.0631, 0.5460), vec3(-0.4776, 0.2847,-0.0271) );

   // grab a normal for reflecting the sample rays later on
   vec3 fres = normalize(texture2D(tex_noise, v_texcoord).xyz*2.0 - 1.0);
   vec4 norm_depth = texture2D(tex_normal_depth, v_texcoord);
   vec3 normal = norm_depth.rgb;
   float depth = norm_depth.a;

   // current fragment coords in screen space
   vec3 ep = vec3(v_texcoord, depth);
   float bl = 0.0;
   // adjust for the depth ( not shure if this is good..)
   float radD = radius / depth;
   vec3 ray, se, occNorm;
   float occluderDepth, depthDifference, normDiff;
   for(int i=0; i<samples; i++)
   {
      // Get a random vector inside the unit sphere
      ray = radD * reflect(sample_kernel[i], fres);
      // If the ray is outside the hemisphere then change direction
      se = ep + sign(dot(ray,normal) )*ray;
      // Get the depth of the occluder fragment
      vec4 occluderFragment = texture2D(tex_normal_depth,se.xy);
      // get the normal of the occluder fragment
      occNorm = occluderFragment.xyz;
      // if depthDifference is negative = occluder is behind current fragment
      depthDifference = depth - occluderFragment.a;
      // calculate the difference between the normals as a weight
      normDiff = (1.0-dot(occNorm,normal));
      // the falloff equation, starts at falloff and is kind of 1/x^2 falling
      bl += step(falloff,depthDifference)*normDiff*(1.0-smoothstep(falloff,strength,depthDifference));
   }
   // output the result
   float ao = 1.0 - base*bl/16;

   vec4 color  = texture2D(tex_color, v_texcoord);
   $out_color = vec4(ao, 0, 0, color.a);
}
"""


copy_frag = """#version 450
uniform sampler2D tex_noise;
uniform sampler2D tex_color;
uniform sampler2D tex_normal_depth;

uniform float base;
uniform float radius;
uniform float falloff;
uniform float strength;

in vec2 v_texcoord;

void main(void) {
   vec4 color = texture2D(tex_color, v_texcoord);
   $out_color = color;
}
"""


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

        width, height = canvas.size

        # A quad (two triangles) spanning the framebuffer area.
        self.screen_vertices = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        # ambient_occlusion
        self.ssao_prog = ModularProgram(vcode=vert_ssao, fcode=copy_frag)
        self.ssao_prog['a_position'] = self.screen_vertices
        self.color_texture = Texture2D((height, width, 4), format='rgba')
        self.ssao_prog['tex_color'] = self.color_texture
        self.normal_depth_texture = Texture2D((height, width, 4), format='rgba')
        self.ssao_prog['tex_normal_depth'] = self.normal_depth_texture

        self.out_color = Variable('out vec4 out_color')
        self.ssao_prog.frag['out_color'] = self.out_color

        self.depth_buffer = RenderBuffer((height, width), 'depth')
        self.framebuffer = FrameBuffer(
            color=[self.color_texture, self.normal_depth_texture],
            depth=self.depth_buffer
        )

        self.noise = np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)
        self.ssao_prog['tex_noise'] = self.noise

        self.ssao_prog['base'] = 1.00
        self.ssao_prog['strength'] = 0.20
        self.ssao_prog['falloff'] = 0.000002
        self.ssao_prog['radius'] = 0.01

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
            depth_test=True,
        )
        push_fbo()
        canvas.context.clear(color=bgcolor)
        canvas.draw_visual(canvas.scene)
        pop_fbo()

        # draw on canvas back buffer
        # canvas.context.set_state(
            # depth_test=True,
            # blend=True,
            # depth_mask=True,
        # )
        # canvas.context.clear(color=bgcolor)
        self.ssao_prog.draw('triangle_strip')
