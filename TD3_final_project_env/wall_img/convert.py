from PIL import Image as im
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

a_img_state_0_0 = im.open('austria_competition_state 0 - 0.png')
c_img_state_0_0 = im.open('circle_cw_competition_collisionStop_state 0 - 0.png')

c_left = im.open('left_wall.png')
c_right = im.open('right_wall.png')

a_img_state_0_0 = np.array(a_img_state_0_0)
c_img_state_0_0 = np.array(c_img_state_0_0)

c_left = np.array(c_left)
c_right = np.array(c_right)

for i in range(128):
    print("[", end="")
    for j in range(128):
        print(c_right[i][j], end="")
        if j != 127:
            print(", ", end="")
    print("], ")


