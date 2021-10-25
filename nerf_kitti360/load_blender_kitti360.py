import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    all_imgs = []
    all_poses = []
    counts = [0]
    cam2world = open("./cam0_to_world.txt", 'r').readlines()
    split_pace = {'train':2, 'val':9, 'test':7}
    for s in splits:
        imgs = []
        poses = []
        for i in range(0, 150, split_pace[s]):
            fname = os.path.join(basedir, '0000000' + str(int(500+i)) + '.png')
            imgs.append(imageio.imread(fname))
            line = list(map(float, cam2world[409+i].strip().split(" ")))
            pose = np.array(line[1:]).reshape(4,4)
            poses.append(pose)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)  
    H, W = imgs[0].shape[:2]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 3970) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_poses = torch.Tensor(np.array([[-0.537911,-7.502336e-02,0.839657,1.075212e+03],
                                        [-0.842868, 6.562258e-02,-0.534104,3.803574e+03],
                                        [-0.015030,-9.950204e-01,-0.098533,1.158312e+02],
                                        [0,0,0,1]]))
    render_poses = torch.stack([render_poses],0)
    focal = 787
    return imgs, poses, render_poses, [H, W, focal], i_split