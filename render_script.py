
#### this script is used to generate synthetic rgb images in blender

import bpy
import math
import random
from random import *
item = bpy.data.objects['Empty']
item.rotation_mode = 'XYZ'
rotate_by = 15
start_angle = 30
counter = 1
random_numbers = [2,0,2,0,2,1,2,0,1]
for x in range(0,5):
    angle = (start_angle * (math.pi/180)) + (x*-1) * (rotate_by * (math.pi/180))  
    
    ran_num = random_numbers[x]
    if ran_num == 0:
        item.rotation_euler = ( angle,0,0 )
    elif ran_num == 1:
        item.rotation_euler = ( 0,angle,0)
    elif ran_num == 2:
        item.rotation_euler = ( 0,0,angle)
    bpy.context.scene.render.filepath="/home/rgupta/Desktop/training_images/real_image%d.png" % (counter)
    counter+=1
    bpy.ops.render.render(write_still=True, use_viewport=True)