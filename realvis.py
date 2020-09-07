from os import chdir
chdir('//172.16.50.230/1803031/acfp')

from open3d import io
from acfp import multi_board, pinhole, color_max, color_type
from numpy import array, ones, save
from pyglet.image import ImageData
from time import time

t = time()
cloud = io.read_point_cloud('tnkz_laser.ply')
print(time() - t)
print(cloud)

t = time()
array_cloud = ones((len(cloud.points), 7))
array_cloud[:, :3] = cloud.points
array_cloud[:, 3:6] = cloud.colors
print(time() - t)

radius_pt = 0.016
board_hd = multi_board(layers=51, depth=0.1, center=2)
n_jobs = 30

center = array((-7.658, -53.870, -100.283))
quat = array((
    0.7025613967450167,
    -0.08004676010739718,
    -0.08004676010739718,
    0.7025613967450167
))
camera = pinhole(540.)

t = time()
board_hd.multi_proj(n_jobs, array_cloud, center, quat, camera, radius_pt)
print(time() - t, n_jobs)

save('tnkz', board_hd.data)

t = time()
image = ImageData(board_hd.width, board_hd.height, 'RGB', bytes(board_hd))
print(time() - t)
image.save('tnkz_ssor.png')

t = time()
image = ImageData(
    board_hd.width, board_hd.height, 'RGB', board_hd.mono_image().tobytes())
print(time() - t)
image.save('depth_ssor.png')

t = time()
image = ImageData(
    board_hd.width - 2, board_hd.height - 2, 'RGB',
    board_hd.mono_image(board_hd.light()).tobytes())
print(time() - t)
image.save('light_ssor.png')

t = time()
image = ImageData(
    board_hd.width - 2, board_hd.height - 2, 'RGB',
    board_hd.mono_image(board_hd.ssao()).tobytes())
print(time() - t)
image.save('ssao_ssor.png')

t = time()
image = ImageData(
    board_hd.width - 2, board_hd.height - 2, 'RGB',
    board_hd.mono_image(board_hd.edl()).tobytes())
print(time() - t)
image.save('edl_ssor.png')

t = time()
image = ImageData(
    board_hd.width, board_hd.height, 'RGB',
    (
        board_hd.data[:, :, 0, :board_hd.fields] * color_max
    ).astype(color_type).tobytes()
)
print(time() - t)
image.save('tnkz_near.png')

print('layer', weight[:, :, :, 0].argmin(2).max())
