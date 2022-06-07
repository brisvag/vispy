# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from ..gloo import VertexBuffer, Texture1D
from ..gloo.texture import should_cast_to_f32
from .visual import Visual

import numpy as np

# Vertex shader
VERT_SHADER = """
attribute vec2 a_position;

varying vec2 v_position;
varying vec4 v_farpos;
varying vec4 v_nearpos;

void main() {
    v_position = a_position;
    vec4 pos_in_cam = $visual_to_framebuffer(vec4(v_position, 0, 1));

    // intersection of ray and near clipping plane (z = -1 in clip coords)
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = $framebuffer_to_visual(pos_in_cam);

    // intersection of ray and far clipping plane (z = +1 in clip coords)
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = $framebuffer_to_visual(pos_in_cam);

    gl_Position = vec4(v_position, 0, 1);
}
"""  # noqa

# Fragment shader
FRAG_SHADER = """
uniform $sampler_type u_atoms;
uniform int u_n_atoms;
uniform vec2 iResolution;

varying vec2 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

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

float raycast(vec3 eye, vec3 view_dir) {
    float t = 0.5;
    for (int i = 0; i < 256 && t < 20; i++) {
        vec3 pos = nearpos + view_ray * t;

        float dist = map(pos);

        if (abs(dist) < 0.0005 * t) {
            res = dist;
            break;
        }
        t += dist;
    }
    return res
}

void main() {
    vec4 col = vec4(0, 0, 0, 0);

    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

    // Calculate unit vector pointing in the view direction through this
    // fragment.
    vec3 view_ray = normalize(farpos.xyz - nearpos.xyz);

    // TODO: account for resolution
    //gl_FragColor = vec4(view_ray, 1);
    //return;

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
        # view.view_program.vert['visual_to_render'] = view.get_transform('visual', 'render')
        view.view_program.vert['visual_to_framebuffer'] = view.get_transform('visual', 'framebuffer')
        view.view_program.vert['framebuffer_to_visual'] = view.get_transform('framebuffer', 'visual')
