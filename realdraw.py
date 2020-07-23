from open3d import io
from ncfp import board, pinhole
from numpy import array
from pyglet.window import Window
from pyglet.image import ImageData
from pyglet.app import run
from time import time

t = time()
cloud = io.read_point_cloud('test.ply')
print(time() - t, 0)
print(cloud)

t = time()
board_hd = board()
center = array((400., 1100., 400.))
quat = array((1., 0., 0., 0.))
print(time() - t, 1)

t = time()
board_hd.proj_o3d(cloud, center, quat, pinhole())
print(time() - t, 2)

t = time()
window = Window(fullscreen=True)
print(time() - t, 3)

t = time()
image = ImageData(board_hd.width, board_hd.height, 'RGB', bytes(board_hd))
print(time() - t, 4)

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0)

run()
