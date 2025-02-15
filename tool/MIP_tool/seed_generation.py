# -*- encoding: utf-8 -*-
'''
@File    :   seed_generation.py    
@Modify Time      @Author       
------------      --------    
13/10/2024 09:30   zhengyang             

@Desciption
-----------
None
'''

def seed_generation_function(args, highres_cam):
    Tfg = args.Tfg
    Tbg = args.Tbg

    highres_cam[highres_cam<Tfg]=0
    return highres_cam