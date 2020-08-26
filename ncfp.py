from numpy import zeros, array, concatenate, log2, seterr, fmax, where
from scipy.spatial.transform import Rotation
from sys import float_info
from joblib import Parallel, delayed

seterr(divide='ignore', invalid='ignore')

_width = 1920
_height = 1080
exponent_decay = -15
depth = 0.05
color_max = 255.9
color_type = 'uint8'

#_cross_sec = lambda r: (2. * r - 3.) * r ** 2 + 1.
_cross_sec = lambda r: 1. - r

log2sum = lambda a, b: fmax(a, b) + where(
    a == b, 1., log2(1. + 0.5 ** abs(a - b)))

def pinhole(f=_width/2., px=_width/2., py=_height/2.):
    return array((
        (-f, 0., px),
        (0., f, py),
        (0., 0., 1.)
    ))

class board(object):
    
    def __init__(
        self, width=_width, height=_height, fields=3, dtype=float,
        back_weight=float_info.min, cross_sec=_cross_sec, near_clip=1.,
        far_clip=float('inf'), radius_min=0.8, radius_max=float('inf'),
        *args, **kwargs
    ):
        self.width = width = max(int(width), 1)
        self.height = height = max(int(height), 1)
        self.fields = fields = max(int(fields), 0)
        self.dtype = dtype
        self.data = zeros((height, width, fields + 2), dtype)
        self.back_weight = max(float(back_weight), 0.)
        self.cross_sec = cross_sec
        self.near_clip = near_clip = max(float(near_clip), 0.)
        self.far_clip = far_clip = max(float(far_clip), near_clip)
        self.radius_min = radius_min = max(float(radius_min), 0.)
        self.radius_max = radius_max = max(float(radius_max), radius_min)
    
    def image(self):
        return (
            self.data[:, :, :self.fields] /
            fmax(self.data[:, :, -1:], self.back_weight) *
            color_max
        ).astype(color_type)
    
    def __bytes__(self):
        return self.image().tobytes()
    
    def depth(self):
        data_depth = zeros((self.height, self.width, self.fields), color_type)
        data_depth[:, :] = (
            self.data[:, :, self.fields:self.fields+1] /
            fmax(self.data[:, :, -1:], self.back_weight) *
            color_max
        )
        return data_depth
    
    def clear(self):
        self.data[:] = 0.
    
    def merge(self, value):
        self.data += value
    
    def draw_pix(self, x, y, r, dist, color, decay, *args, **kwargs):
        weight = self.cross_sec(r) * decay
        self.data[y, x, :self.fields] += color * weight
        self.data[y, x, self.fields] += 1. / dist * weight
        self.data[y, x, -1] += weight
    
    def draw_quad(self, xi, yi, dx, dy, x0, y0, radius, *args, **kwargs):
        x = xi
        y = yi
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        while (r < radius or x != xi) and 0 <= y < self.height:
            if r < radius and 0 <= x < self.width:
                self.draw_pix(x, y, r / radius, *args, **kwargs)
                x += dx
            else:
                x = xi
                y += dy
            r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
    
    def draw(self, x0, y0, dist, color, alpha, size_pt):
        dist = min(max(dist, self.near_clip), self.far_clip)
        radius = min(max(size_pt / dist, self.radius_min), self.radius_max)
        decay = alpha * dist ** exponent_decay
        x0i = int(x0)
        y0i = int(y0)
        x0 -= 0.5
        y0 -= 0.5
        self.draw_quad(x0i, y0i, 1, 1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i, y0i-1, 1, -1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i-1, y0i, -1, 1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i-1, y0i-1, -1, -1, x0, y0, radius, dist, color, decay)
    
    def proj_pt(self, pt, center, rotation, camera, size_pt):
        v_cam = rotation @ (pt[:3] - center)
        if 0. < v_cam[2]:
            v = camera @ v_cam
            x0 = v[0] / v[2]
            y0 = v[1] / v[2]
            if 0. <= x0 < self.width and 0. <= y0 < self.height:
                self.draw(
                    x0, y0, (v_cam @ v_cam) ** 0.5, pt[3:self.fields+3], pt[-1],
                    size_pt)
    
    def proj(self, cloud, center, quat, camera, radius_pt, *args, **kwargs):
        rotation = Rotation(quat).inv().as_matrix()
        size_pt = radius_pt * camera[1, 1]
        {self.proj_pt(pt, center, rotation, camera, size_pt) for pt in cloud}
    
    def temp_proj(self, *args, **kwargs):
        board_temp = type(self)(**self.__dict__)
        board_temp.proj(*args, **kwargs)
        return board_temp
    
    def multi_proj(self, n_jobs, cloud, *args, **kwargs):
        n_jobs = max(int(n_jobs), 1)
        for board_temp in Parallel(n_jobs)(delayed(self.temp_proj)(
            cloud[i::n_jobs], *args, **kwargs
        ) for i in range(n_jobs)):
            self.merge(board_temp.data)

class board_near(board):
    
    def merge(self, value):
        self.data[:] = where(
            self.data[:, :, self.fields:self.fields+1]
            >= value[:, :, self.fields:self.fields+1],
            self.data, value
        )
    
    def draw_pix(self, x, y, r, dist, color, *args, **kwargs):
        if self.data[y, x, self.fields] < 1. / dist:
            self.data[y, x, :self.fields] = color
            self.data[y, x, self.fields] = 1. / dist
            self.data[y, x, -1] = 1.

class log2board(board):
    
    def __init__(self, back_weight=-float_info.max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data[:] = float('-inf')
        self.back_weight = float(back_weight)
    
    def image(self):
        return (2 ** (
            self.data[:, :, :self.fields] -
            fmax(self.data[:, :, -1:], self.back_weight)
        ) * color_max).astype(color_type)
    
    def depth(self):
        data_depth = zeros((self.height, self.width, self.fields), color_type)
        data_depth[:, :] = (2 ** (
            self.data[:, :, self.fields:self.fields+1] -
            fmax(self.data[:, :, -1:], self.back_weight)
        ) * color_max)
        return data_depth
    
    def clear(self):
        self.data[:] = float('-inf')
    
    def merge(self, value):
        self.data[:] = log2sum(self.data, value)
    
    def draw_pix(self, x, y, r, dist, color, decay, *args, **kwargs):
        weight = log2(self.cross_sec(r)) + decay
        self.data[y, x, :self.fields] = log2sum(
            self.data[y, x, :self.fields], weight + color)
        self.data[y, x, self.fields] = log2sum(
            self.data[y, x, self.fields], weight - dist)
        self.data[y, x, -1] = log2sum(self.data[y, x, -1], weight)
    
    def draw(self, x0, y0, dist, color, alpha, size_pt):
        dist = min(max(dist, self.near_clip), self.far_clip)
        radius = min(max(size_pt / dist, self.radius_min), self.radius_max)
        decay = log2(alpha) - dist / depth
        dist = log2(dist)
        color = log2(color)
        x0i = int(x0)
        y0i = int(y0)
        x0 -= 0.5
        y0 -= 0.5
        self.draw_quad(x0i, y0i, 1, 1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i, y0i-1, 1, -1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i-1, y0i, -1, 1, x0, y0, radius, dist, color, decay)
        self.draw_quad(x0i-1, y0i-1, -1, -1, x0, y0, radius, dist, color, decay)
