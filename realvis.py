from open3d import io
from acfp import multi_board, pinhole
from numpy import array, ones
from pyglet.image import ImageData
from pyglet.window import Window
from pyglet.app import run

cloud = io.read_point_cloud('test.ply')
print(cloud)
array_cloud = ones((len(cloud.points), 7))
array_cloud[:, :3] = cloud.points
array_cloud[:, 3:6] = cloud.colors
board_hd = multi_board(radius_min=1.6, radius_max=40., depth=0.4)

center = array((475., 1100., 50.))
quat = array((0.70710678, 0., 0., 0.70710678))
camera = pinhole()
radius_pt = 0.08
n_jobs = 10

board_hd.multi_proj(n_jobs, array_cloud, center, quat, camera, radius_pt)
image = ImageData(board_hd.width, board_hd.height, 'RGB', bytes(board_hd))
image.save('test.png')

window = Window(fullscreen=True)

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0)

run()
