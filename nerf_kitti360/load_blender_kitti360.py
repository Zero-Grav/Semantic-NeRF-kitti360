import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    all_imgs = []
    all_poses = []
    counts = [0]
    cam2world = {}
    start = 505
    num = 1
    split_pace = {'train':1, 'val':1, 'test':1}
    for line in open("./cam0_to_world.txt", 'r').readlines():
        value = list(map(float, line.strip().split(" ")))
        cam2world[value[0]] = np.array(value[1:]).reshape(4,4)
    start_inverse = np.linalg.inv(cam2world[start])
    for s in splits:
        imgs = []
        poses = []
        for i in range(start, start+num, split_pace[s]):
            # if s == 'train' and i % (split_pace['val']-start) ==0:
            #     continue
            fname = os.path.join(basedir, '0000000' + str(int(i)) + '.png')
            imgs.append(imageio.imread(fname))
            if i == start:
                pose = np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]])
            else:
                pose = np.dot(start_inverse, cam2world[i])
            poses.append(pose)
        imgs = (np.array(imgs)/255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)  
    H, W = imgs[0].shape[:2]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    render_poses = np.dot(start_inverse, cam2world[start])
    render_poses = torch.stack([torch.Tensor(render_poses)],0)
    focal = 552.554261
    return imgs, poses, render_poses, [H, W, focal], i_split