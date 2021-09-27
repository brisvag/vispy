# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from ..gloo import VertexBuffer, Texture1D
from ..gloo.texture import should_cast_to_f32
from .visual import Visual

import numpy as np

# todo: implement more render methods (port from visvis)
# todo: allow anisotropic data
# todo: what to do about lighting? ambi/diffuse/spec/shinynes on each visual?


# Vertex shader
VERT_SHADER = """
attribute vec2 a_position;

varying vec2 v_position;
varying vec3 v_view_dir;

void main() {
    gl_Position = vec4(a_position, 0, 1);

    // calculate view direction accound for perspective
    vec4 pos_visual = $render_to_visual(gl_Position);
    pos_visual /= pos_visual.w;
    vec4 pos_front = pos_visual;
    vec4 pos_back = pos_visual;
    pos_front.z -= 1e-5;
    pos_back.z += 1e-5;
    pos_front = $visual_to_render(pos_front);
    pos_back = $visual_to_render(pos_back);

    v_view_dir = normalize(pos_back.xyz / pos_back.w - pos_front.xyz / pos_front.w);

    v_position = a_position;
}
"""  # noqa

# Fragment shader
FRAG_SHADER = """
uniform $sampler_type u_atoms;
uniform int u_n_atoms;
uniform vec2 iResolution;

varying vec2 v_position;

float sdf_sphere(vec3 pos, vec3 sphere_pos) {
    float d = length(pos - sphere_pos) - 0.1;
    return d;
}

float map(vec3 pos) {
    float d = 100000;
    for (int i = 0; i < u_n_atoms; i++) {
        vec4 atom_pos = $sample(u_atoms, i);
        d = min(d, sdf_sphere(pos, atom_pos.xyz));
    }
    return d;
}

void main() {
    vec4 ro = $render_to_visual(vec4(0, 0, 1, 0));
    vec4 rd = normalize($render_to_visual(vec4(0, 0, -1, 0)));

    vec4 col = vec4(0, 0, 0, 1);

    // TODO: account for resolution

    float t = 0.0;
    for (int i = 0; i < 100; i++) {
        vec3 pos = ro.xyz + t * rd.xyz;

        float h = map(pos);

        if (h < 0.001)
            break;

        t += h;
        if (t > 20.0)
            discard;
    }

    if (t <= 20) {
        col = vec4(1, 1, 1, 1);
    }

    gl_FragColor = col;
}


"""  # noqa


class MoleculeVisual(Visual):
    def __init__(self, atoms):
        # Create gloo objects
        self._vertices = VertexBuffer(np.array([
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
        ], dtype=np.float32))
        self._atoms = Texture1D(atoms)

        # Create program
        Visual.__init__(self, vcode=VERT_SHADER, fcode=FRAG_SHADER)
        self.shared_program['u_atoms'] = self._atoms
        self.shared_program['u_n_atoms'] = len(atoms)
        self.shared_program['a_position'] = self._vertices
        self._draw_mode = 'triangle_strip'

        self.shared_program.frag['sampler_type'] = self._atoms.glsl_sampler_type
        self.shared_program.frag['sample'] = self._atoms.glsl_sample

        self.set_gl_state('translucent')

        self.set_data(atoms)
        self.freeze()

    def set_data(self, atoms, copy=True):
        # Check volume
        if not isinstance(atoms, np.ndarray):
            raise ValueError('atomsume visual needs a numpy array.')
        if isinstance(self._atoms, Texture1D):
            copy = False

        if should_cast_to_f32(atoms.dtype):
            atoms = atoms.astype(np.float32)

        self._atoms.set_data(atoms, copy=copy)  # will be efficient if atoms is same shape

    def _compute_bounds(self, axis, view):
        return None

    def _prepare_transforms(self, view):
        view.view_program.frag['render_to_visual'] = view.get_transform('render', 'visual')
