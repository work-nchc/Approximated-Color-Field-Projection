from numpy import zeros, array, concatenate
from scipy.spatial.transform import Rotation
from sys import float_info

with open('ncfp.cfg') as cfg_file:
    for cfg in cfg_file:
        parse = cfg.split()
        if 2 == len(parse):
            if '_width' == parse[0]:
                _width = int(parse[-1])
            if '_height' == parse[0]:
                _height = int(parse[-1])
            if '_fields' == parse[0]:
                _fields = int(parse[-1])
            if 'exponent' == parse[0]:
                exponent = int(parse[-1])
            if 'near_clip' == parse[0]:
                near_clip = float(parse[-1])
            if 'radius_min' == parse[0]:
                radius_min = float(parse[-1])
            if 'color_max' == parse[0]:
                color_max = float(parse[-1])

cross_sec = lambda r: (2. * r - 3.) * r ** 2 + 1.

def pinhole(f=_width/2., px=_width/2., py=_height/2.):
    return array((
        (-f, 0., px),
        (0., f, py),
        (0., 0., 1.)
    ))

class board(object):
    
    def __init__(
        self, width=_width, height=_height, fields=_fields, dtype=float
    ):
        self.width = width = max(int(width), 1)
        self.height = height = max(int(height), 1)
        self.fields = fields = max(int(fields), 0)
        self.data = zeros((height, width, fields + 2), dtype)
        self.data[:, :, -1].fill(float_info.min)
    
    def __iadd__(self, value):
        self.data += value.data
    
    def image(self):
        return (
            self.data[:, :, :self.fields] / self.data[:, :, -1:] * color_max
        ).astype('uint8')
    
    def __bytes__(self):
        return self.image().tobytes()
    
    def draw_pix(self, x, y, r, dist, color, alpha):
        weight = alpha * cross_sec(r) * dist ** exponent
        self.data[y, x, :self.fields] += color * weight
        self.data[y, x, self.fields] += dist * weight
        self.data[y, x, -1] += weight
    
    def draw_quad(self, xi, yi, dx, dy, x0, y0, radius, dist, color, alpha):
        x = xi
        y = yi
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        while (r < radius or x != xi) and 0 <= y < self.height:
            if r < radius and 0 <= x < self.width:
                self.draw_pix(x, y, r / radius, dist, color, alpha)
                x += dx
            else:
                x = xi
                y += dy
            r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
    
    def draw(self, x0, y0, dist, color, alpha, camera, radius_pt):
        dist = max(dist, near_clip)
        radius = max(radius_pt * camera[1, 1] / dist, radius_min)
        x0i = int(x0)
        y0i = int(y0)
        x0 -= 0.5
        y0 -= 0.5
        self.draw_quad(x0i, y0i, 1, 1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i, y0i-1, 1, -1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i-1, y0i, -1, 1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i-1, y0i-1, -1, -1, x0, y0, radius, dist, color, alpha)
    
    def proj_pt(self, pt, center, quat, camera, radius_pt):
        v_cam = Rotation(quat).apply(pt[:3] - center, True)
        if 0. < v_cam[2]:
            v = camera @ v_cam
            x0 = v[0] / v[2]
            y0 = v[1] / v[2]
            if 0. <= x0 < self.width and 0. <= y0 < self.height:
                self.draw(
                    x0, y0, (v_cam @ v_cam) ** 0.5, pt[3:self.fields+3], pt[-1],
                    camera, radius_pt)
    
    def proj(self, cloud, center, quat, camera, radius_pt):
        {self.proj_pt(pt, center, quat, camera, radius_pt) for pt in cloud}
    
    def proj_o3d(self, cloud, center, quat, camera, radius_pt):
        {
            self.proj_pt(
                concatenate((*pt, array((1.,)))), center, quat, camera,
                radius_pt
            ) for pt in zip(cloud.points, cloud.colors)
        }
