# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""Primitive 2D image visual class."""

from __future__ import division

import contextlib

import numpy as np

from ..gloo import Texture2D, VertexBuffer
from ..gloo.texture import should_cast_to_f32
from ..color import get_colormap
from .shaders import Function, FunctionChain
from .transforms import NullTransform
from .visual import Visual
from ..io import load_spatial_filters
from ._scalable_textures import CPUScaledTexture2D, GPUScaledTexture2D


_VERTEX_SHADER = """
uniform int method;  // 0=subdivide, 1=impostor
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position, 0., 1.));
}
"""

_FRAGMENT_SHADER = """
uniform vec2 image_size;
uniform int method;  // 0=subdivide, 1=impostor
uniform sampler2D u_texture;
varying vec2 v_texcoord;

vec4 map_local_to_tex(vec4 x) {
    // Cast ray from 3D viewport to surface of image
    // (if $transform does not affect z values, then this
    // can be optimized as simply $transform.map(x) )
    vec4 p1 = $transform(x);
    vec4 p2 = $transform(x + vec4(0, 0, 0.5, 0));
    p1 /= p1.w;
    p2 /= p2.w;
    vec4 d = p2 - p1;
    float f = p2.z / d.z;
    vec4 p3 = p2 - d * f;

    // finally map local to texture coords
    return vec4(p3.xy / image_size, 0, 1);
}


void main()
{
    vec2 texcoord;
    if( method == 0 ) {
        texcoord = v_texcoord;
    }
    else {
        // vertex shader outputs clip coordinates;
        // fragment shader maps to texture coordinates
        texcoord = map_local_to_tex(vec4(v_texcoord, 0, 1)).xy;
    }

    gl_FragColor = $color_transform($get_data(texcoord));
}
"""  # noqa

_INTERPOLATION_TEMPLATE = """
    #include "misc/spatial-filters.frag"
    vec4 texture_lookup_filtered(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        return %s($texture, $shape, texcoord);
    }"""

_TEXTURE_LOOKUP = """
    vec4 texture_lookup(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        return texture2D($texture, texcoord);
    }"""

_APPLY_CLIM_FLOAT = """
    float apply_clim(float data) {
        // If data is NaN, don't draw it at all
        // http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
        if (!(data <= 0.0 || 0.0 <= data)) {
            discard;
        }
        data = clamp(data, min($clim.x, $clim.y), max($clim.x, $clim.y));
        data = (data - $clim.x) / ($clim.y - $clim.x);
        return data;
    }"""

_APPLY_CLIM = """
    vec4 apply_clim(vec4 color) {
        // Handle NaN values
        // http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
        color.r = !(color.r <= 0.0 || 0.0 <= color.r) ? min($clim.x, $clim.y) : color.r;
        color.g = !(color.g <= 0.0 || 0.0 <= color.g) ? min($clim.x, $clim.y) : color.g;
        color.b = !(color.b <= 0.0 || 0.0 <= color.b) ? min($clim.x, $clim.y) : color.b;
        color.a = !(color.a <= 0.0 || 0.0 <= color.a) ? 0 : color.a;
        color.rgb = clamp(color.rgb, min($clim.x, $clim.y), max($clim.x, $clim.y));
        color.rgb = (color.rgb - $clim.x) / ($clim.y - $clim.x);
        return max(color, 0.0);
    }
"""

_APPLY_GAMMA_FLOAT = """
    float apply_gamma(float data) {
        return pow(data, $gamma);
    }"""

_APPLY_GAMMA = """
    vec4 apply_gamma(vec4 color) {
        color.rgb = pow(color.rgb, vec3($gamma));
        return color;
    }
"""

_NULL_COLOR_TRANSFORM = 'vec4 pass(vec4 color) { return color; }'

_C2L_RED = 'float cmap(vec4 color) { return color.r; }'


class ImageVisual(Visual):
    """Visual subclass displaying an image.

    Parameters
    ----------
    data : ndarray
        ImageVisual data. Can be shape (M, N), (M, N, 3), or (M, N, 4).
        If floating point data is provided and contains NaNs, they will
        be made transparent (discarded) for the single band data case when
        scaling is done on the GPU (see ``texture_format``). On the CPU,
        single band NaNs are mapped to 0 as they are sent to the GPU which
        result in them using the lowest ``clim`` value in the GPU.
        For RGB data, NaNs will be mapped to the lowest ``clim`` value.
        If the Alpha band is NaN it will be mapped to 0 (transparent).
        Note that NaN handling is not required by some OpenGL implementations
        and NaNs may be treated differently on some systems (ex. as 0s).
    method : str
        Selects method of rendering image in case of non-linear transforms.
        Each method produces similar results, but may trade efficiency
        and accuracy. If the transform is linear, this parameter is ignored
        and a single quad is drawn around the area of the image.

            * 'auto': Automatically select 'impostor' if the image is drawn
              with a nonlinear transform; otherwise select 'subdivide'.
            * 'subdivide': ImageVisual is represented as a grid of triangles
              with texture coordinates linearly mapped.
            * 'impostor': ImageVisual is represented as a quad covering the
              entire view, with texture coordinates determined by the
              transform. This produces the best transformation results, but may
              be slow.

    grid: tuple (rows, cols)
        If method='subdivide', this tuple determines the number of rows and
        columns in the image grid.
    cmap : str | ColorMap
        Colormap to use for luminance images.
    clim : str | tuple
        Limits to use for the colormap. I.e. the values that map to black and white
        in a gray colormap. Can be 'auto' to auto-set bounds to
        the min and max of the data. If not given or None, 'auto' is used.
    gamma : float
        Gamma to use during colormap lookup.  Final color will be cmap(val**gamma).
        by default: 1.
    interpolation : str
        Selects method of image interpolation. Makes use of the two Texture2D
        interpolation methods and the available interpolation methods defined
        in vispy/gloo/glsl/misc/spatial_filters.frag

            * 'nearest': Default, uses 'nearest' with Texture2D interpolation.
            * 'bilinear': uses 'linear' with Texture2D interpolation.
            * 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'bicubic',
                'catrom', 'mitchell', 'spline16', 'spline36', 'gaussian',
                'bessel', 'sinc', 'lanczos', 'blackman'
    texture_format: numpy.dtype | str | None
        How to store data on the GPU. OpenGL allows for many different storage
        formats and schemes for the low-level texture data stored in the GPU.
        Most common is unsigned integers or floating point numbers.
        Unsigned integers are the most widely supported while other formats
        may not be supported on older versions of OpenGL or with older GPUs.
        Default value is ``None`` which means data will be scaled on the
        CPU and the result stored in the GPU as an unsigned integer. If a
        numpy dtype object, an internal texture format will be chosen to
        support that dtype and data will *not* be scaled on the CPU. Not all
        dtypes are supported. If a string, then
        it must be one of the OpenGL internalformat strings described in the
        table on this page: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        The name should have `GL_` removed and be lowercase (ex.
        `GL_R32F` becomes ``'r32f'``). Lastly, this can also be the string
        ``'auto'`` which will use the data type of the provided image data
        to determine the internalformat of the texture.
        When this is specified (not ``None``) data is scaled on the
        GPU which allows for faster color limit changes. Additionally, when
        32-bit float data is provided it won't be copied before being
        transferred to the GPU.
    **kwargs : dict
        Keyword arguments to pass to `Visual`.

    Notes
    -----
    The colormap functionality through ``cmap`` and ``clim`` are only used
    if the data are 2D.
    """

    _shaders = {
        'vertex': _VERTEX_SHADER,
        'fragment': _FRAGMENT_SHADER,
    }

    _func_templates = {
        'texture_lookup_interpolated': _INTERPOLATION_TEMPLATE,
        'texture_lookup': _TEXTURE_LOOKUP,
        'clim_float': _APPLY_CLIM_FLOAT,
        'clim': _APPLY_CLIM,
        'gamma_float': _APPLY_GAMMA_FLOAT,
        'gamma': _APPLY_GAMMA,
        'null_color_transform': _NULL_COLOR_TRANSFORM,
        'red_to_luminance': _C2L_RED,
    }

    def __init__(self, data=None, method='auto', grid=(1, 1),
                 cmap='viridis', clim='auto', gamma=1.0,
                 interpolation='nearest', texture_format=None, **kwargs):
        """Initialize image properties, texture storage, and interpolation methods."""
        self._data = None

        # load 'float packed rgba8' interpolation kernel
        # to load float interpolation kernel use
        # `load_spatial_filters(packed=False)`
        kernel, interpolation_names = load_spatial_filters()

        self._kerneltex = Texture2D(kernel, interpolation='nearest')
        # The unpacking can be debugged by changing "spatial-filters.frag"
        # to have the "unpack" function just return the .r component. That
        # combined with using the below as the _kerneltex allows debugging
        # of the pipeline
        # self._kerneltex = Texture2D(kernel, interpolation='linear',
        #                             internalformat='r32f')

        interpolation_names, interpolation_fun = self._init_interpolation(
            interpolation_names)
        self._interpolation_names = interpolation_names
        self._interpolation_fun = interpolation_fun
        self._interpolation = interpolation
        if self._interpolation not in self._interpolation_names:
            raise ValueError("interpolation must be one of %s" %
                             ', '.join(self._interpolation_names))

        self._method = method
        self._grid = grid
        self._need_texture_upload = True
        self._need_vertex_update = True
        self._need_colortransform_update = True
        self._need_interpolation_update = True
        self._texture = self._init_texture(data, texture_format)
        self._subdiv_position = VertexBuffer()
        self._subdiv_texcoord = VertexBuffer()

        # impostor quad covers entire viewport
        vertices = np.array([[-1, -1], [1, -1], [1, 1],
                             [-1, -1], [1, 1], [-1, 1]],
                            dtype=np.float32)
        self._impostor_coords = VertexBuffer(vertices)
        self._null_tr = NullTransform()

        self._init_view(self)

        Visual.__init__(self, vcode=self._shaders['vertex'], fcode=self._shaders['fragment'])
        self.set_gl_state('translucent', cull_face=False)
        self._draw_mode = 'triangles'

        # define _data_lookup_fn as None, will be setup in
        # self._build_interpolation()
        self._data_lookup_fn = None

        self.clim = clim or "auto"  # None -> "auto"
        self.cmap = cmap
        self.gamma = gamma
        if data is not None:
            self.set_data(data)
        self.freeze()

    def _init_interpolation(self, interpolation_names):
        # create interpolation shader functions for available
        # interpolations
        fun = [Function(self._func_templates['texture_lookup_interpolated'] % n)
               for n in interpolation_names]
        interpolation_names = [n.lower() for n in interpolation_names]

        interpolation_fun = dict(zip(interpolation_names, fun))
        interpolation_names = tuple(sorted(interpolation_names))

        # overwrite "nearest" and "bilinear" spatial-filters
        # with  "hardware" interpolation _data_lookup_fn
        hardware_lookup = Function(self._func_templates['texture_lookup'])
        interpolation_fun['nearest'] = hardware_lookup
        interpolation_fun['bilinear'] = hardware_lookup
        return interpolation_names, interpolation_fun

    def _init_texture(self, data, texture_format):
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        if texture_format is None:
            tex = CPUScaledTexture2D(
                data, interpolation=texture_interpolation)
        else:
            tex = GPUScaledTexture2D(
                data, internalformat=texture_format,
                interpolation=texture_interpolation)
        return tex

    def set_data(self, image):
        """Set the image data.

        Parameters
        ----------
        image : array-like
            The image data.
        texture_format : str or None

        """
        data = np.asarray(image)
        if should_cast_to_f32(data.dtype):
            data = data.astype(np.float32)
        # can the texture handle this data?
        self._texture.check_data_format(data)
        if self._data is None or self._data.shape[:2] != data.shape[:2]:
            # Only rebuild if the size of the image changed
            self._need_vertex_update = True
        self._data = data
        self._need_texture_upload = True

    def view(self):
        """Get the :class:`vispy.visuals.visual.VisualView` for this visual."""
        v = Visual.view(self)
        self._init_view(v)
        return v

    def _init_view(self, view):
        # Store some extra variables per-view
        view._need_method_update = True
        view._method_used = None

    @property
    def clim(self):
        """Get color limits used when rendering the image (cmin, cmax)."""
        return self._texture.clim

    @clim.setter
    def clim(self, clim):
        if self._texture.set_clim(clim):
            self._need_texture_upload = True
        self._update_colortransform_clim()
        self.update()

    def _update_colortransform_clim(self):
        if self._need_colortransform_update:
            # we are going to rebuild anyway so just do it later
            return
        try:
            norm_clims = self._texture.clim_normalized
        except RuntimeError:
            return
        else:
            # shortcut so we don't have to rebuild the whole color transform
            self.shared_program.frag['color_transform'][1]['clim'] = norm_clims

    @property
    def cmap(self):
        """Get the colormap object applied to luminance (single band) data."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = get_colormap(cmap)
        self._need_colortransform_update = True
        self.update()

    @property
    def gamma(self):
        """Get the gamma used when rendering the image."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Set gamma used when rendering the image."""
        if value <= 0:
            raise ValueError("gamma must be > 0")
        self._gamma = float(value)
        # shortcut so we don't have to rebuild the color transform
        if not self._need_colortransform_update:
            self.shared_program.frag['color_transform'][2]['gamma'] = self._gamma
        self.update()

    @property
    def method(self):
        """Get rendering method name."""
        return self._method

    @method.setter
    def method(self, m):
        if self._method != m:
            self._method = m
            self._need_vertex_update = True
            self.update()

    @property
    def size(self):
        """Get size of the image (width, height)."""
        return self._data.shape[:2][::-1]

    @property
    def interpolation(self):
        """Get interpolation algorithm name."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, i):
        if i not in self._interpolation_names:
            raise ValueError("interpolation must be one of %s" %
                             ', '.join(self._interpolation_names))
        if self._interpolation != i:
            self._interpolation = i
            self._need_interpolation_update = True
            self.update()

    @property
    def interpolation_functions(self):
        """Get names of possible interpolation methods."""
        return self._interpolation_names

    # The interpolation code could be transferred to a dedicated filter
    # function in visuals/filters as discussed in #1051
    def _build_interpolation(self):
        """Rebuild the _data_lookup_fn for different interpolations."""
        interpolation = self._interpolation
        self._data_lookup_fn = self._interpolation_fun[interpolation]
        self.shared_program.frag['get_data'] = self._data_lookup_fn

        # only 'bilinear' uses 'linear' texture interpolation
        if interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            # 'nearest' (and also 'bilinear') doesn't use spatial_filters.frag
            # so u_kernel and shape setting is skipped
            texture_interpolation = 'nearest'
            if interpolation != 'nearest':
                self.shared_program['u_kernel'] = self._kerneltex
                self._data_lookup_fn['shape'] = self._data.shape[:2][::-1]

        if self._texture.interpolation != texture_interpolation:
            self._texture.interpolation = texture_interpolation

        self._data_lookup_fn['texture'] = self._texture

        self._need_interpolation_update = False

    def _build_vertex_data(self):
        """Rebuild the vertex buffers for the subdivide method."""
        grid = self._grid
        w = 1.0 / grid[1]
        h = 1.0 / grid[0]

        quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0],
                         [0, 0, 0], [w, h, 0], [0, h, 0]],
                        dtype=np.float32)
        quads = np.empty((grid[1], grid[0], 6, 3), dtype=np.float32)
        quads[:] = quad

        mgrid = np.mgrid[0.:grid[1], 0.:grid[0]].transpose(1, 2, 0)
        mgrid = mgrid[:, :, np.newaxis, :]
        mgrid[..., 0] *= w
        mgrid[..., 1] *= h

        quads[..., :2] += mgrid
        tex_coords = quads.reshape(grid[1]*grid[0]*6, 3)
        tex_coords = np.ascontiguousarray(tex_coords[:, :2])
        vertices = tex_coords * self.size

        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))
        self._need_vertex_update = False

    def _update_method(self, view):
        """Decide which method to use for *view* and configure it accordingly."""
        method = self._method
        if method == 'auto':
            if view.transforms.get_transform().Linear:
                method = 'subdivide'
            else:
                method = 'impostor'
        view._method_used = method

        if method == 'subdivide':
            view.view_program['method'] = 0
            view.view_program['a_position'] = self._subdiv_position
            view.view_program['a_texcoord'] = self._subdiv_texcoord
        elif method == 'impostor':
            view.view_program['method'] = 1
            view.view_program['a_position'] = self._impostor_coords
            view.view_program['a_texcoord'] = self._impostor_coords
        else:
            raise ValueError("Unknown image draw method '%s'" % method)

        self.shared_program['image_size'] = self.size
        view._need_method_update = False
        self._prepare_transforms(view)

    def _build_texture(self):
        pre_internalformat = self._texture.internalformat
        self._texture.scale_and_set_data(self._data)
        # color transform needs rebuilding if the internalformat was changed
        # otherwise just update the whole color transform clims
        if self._texture.internalformat != pre_internalformat:
            self._need_colortransform_update = True
        else:
            with contextlib.suppress(KeyError):  # if the shared_program isn't created yet.
                self.shared_program.frag['color_transform'][1]['clim'] = self._texture.clim_normalized
        self._need_texture_upload = False

    def _compute_bounds(self, axis, view):
        if axis > 1:
            return 0, 0
        else:
            return 0, self.size[axis]

    def _build_color_transform(self):
        if self._data.ndim == 2 or self._data.shape[2] == 1:
            # luminance data
            fclim = Function(self._func_templates['clim_float'])
            fgamma = Function(self._func_templates['gamma_float'])
            # NOTE: red_to_luminance only uses the red component, fancy internalformats
            #   may need to use the other components or a different function chain
            fun = FunctionChain(
                None, [Function(self._func_templates['red_to_luminance']), fclim, fgamma, Function(self.cmap.glsl_map)]
            )
        else:
            # RGB/A image data (no colormap)
            fclim = Function(self._func_templates['clim'])
            fgamma = Function(self._func_templates['gamma'])
            fun = FunctionChain(None, [Function(self._func_templates['null_color_transform']), fclim, fgamma])
        fclim['clim'] = self._texture.clim_normalized
        fgamma['gamma'] = self.gamma
        return fun

    def _prepare_transforms(self, view):
        trs = view.transforms
        prg = view.view_program
        method = view._method_used
        if method == 'subdivide':
            prg.vert['transform'] = trs.get_transform()
            prg.frag['transform'] = self._null_tr
        else:
            prg.vert['transform'] = self._null_tr
            prg.frag['transform'] = trs.get_transform().inverse

    def _prepare_draw(self, view):
        if self._data is None:
            return False

        if self._need_interpolation_update:
            self._build_interpolation()

        if self._need_texture_upload:
            self._build_texture()

        if self._need_colortransform_update:
            prg = view.view_program
            self.shared_program.frag['color_transform'] = self._build_color_transform()
            self._need_colortransform_update = False
            prg['texture2D_LUT'] = self.cmap.texture_lut()

        if self._need_vertex_update:
            self._build_vertex_data()

        if view._need_method_update:
            self._update_method(view)
