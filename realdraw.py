from open3d import io
from ncfp import board, pinhole
from numpy import array, save
from pyglet.window import Window
from pyglet.image import ImageData
from pyglet.app import run
from PIL.Image import fromarray
from time import time

t = time()
cloud = io.read_point_cloud('test.ply')
print(time() - t, 0)
print(cloud)

board_hd = board()
#center = array((400., 1100., 400.))
#quat = array((1., 0., 0., 0.))
center = array((475., 1100., 50.))
quat = array((0.70710678, 0., 0., 0.70710678))

t = time()
board_hd.proj_o3d(cloud, center, quat, pinhole(), 0.08)
print(time() - t, 1)

save('test', board_hd.data)

t = time()
img = fromarray(board_hd.image()[::-1], 'RGB')
print(time() - t, 2)

img.save('test.png')

t = time()
window = Window(fullscreen=True)
print(time() - t, 3)

t = time()
image = ImageData(board_hd.width, board_hd.height, 'RGB', bytes(board_hd))
print(time() - t, 4)

image.save('test2.png')

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0)

run()
