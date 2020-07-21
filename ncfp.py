from numpy import zeros, array
from scipy.spatial.transform import Rotation

with open('ncfp.cfg') as cfg_file:
    for cfg in cfg_file:
        parse = cfg.split()
        if 2 == len(parse):
            if '_width' == parse[0]:
                _width = int(parse[-1])
            if '_height' == parse[0]:
                _height = int(parse[-1])
            if 'base' == parse[0]:
                base = float(parse[-1])
            if 'depth' == parse[0]:
                depth = float(parse[-1])
            if 'radius_pt' == parse[0]:
                radius_pt = float(parse[-1])
            if 'near_clip' == parse[0]:
                near_clip = float(parse[-1])
            if 'radius_min' == parse[0]:
                radius_min = float(parse[-1])
            if 'fields' == parse[0]:
                fields = int(parse[-1])

decay = lambda dist: base ** (dist / depth)
cross_sec = lambda r: (2. * r - 3.) * r ** 2 + 1.
radius_dist = lambda dist, camera: radius_pt * camera[0, 0] / dist

def pinhole(f=_width/2., px=_width/2., py=_height/2.):
    return array((
        (f, 0., px),
        (0., f, py),
        (0., 0., 1.)))

class board(object):
    
    def __init__(self, width=_width, height=_height, dtype=float):
        self.width = width = int(width)
        self.height = height = int(height)
        self.data = zeros((height, width, fields + 1), dtype)
    
    def __iadd__(self, value):
        self.data += value.data
    
    def __str__(self):
        return ''.join(map(
            chr,
            map(
                round,
                (self.data[:, :, :3] / self.data[:, :, -1:]).flatten() * 255)))
    
    def draw_pix(self, x, y, r, dist, color, alpha):
        weight = alpha * decay(dist) * cross_sec(r)
        self.data[y, x, :3] += color * weight
        self.data[y, x, 3] += dist * weight
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
    
    def draw(self, x0, y0, dist, color, alpha, camera):
        dist = max(dist, near_clip)
        radius = max(radius_dist(dist, camera), radius_min)
        x0i = int(x0)
        y0i = int(y0)
        x0 -= 0.5
        y0 -= 0.5
        self.draw_quad(x0i, y0i, 1, 1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i, y0i-1, 1, -1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i-1, y0i, -1, 1, x0, y0, radius, dist, color, alpha)
        self.draw_quad(x0i-1, y0i-1, -1, -1, x0, y0, radius, dist, color, alpha)
    
    def proj_pt(self, pt, center, quat, camera):
        v_cam = Rotation(quat).apply(pt[:3] - center)
        if 0. < v_cam[2]:
            v = camera.dot(v_cam)
            x0 = v[0] / v[2]
            y0 = v[1] / v[2]
            if 0. <= x0 < self.width and 0. <= y0 < self.height:
                self.draw(x0, y0, v_cam.dot(v_cam)**0.5, pt[3:6], pt[6], camera)
    
    def proj(self, cloud, center, quat, camera):
        for pt in cloud:
            self.proj_pt(pt, center, quat, camera)
