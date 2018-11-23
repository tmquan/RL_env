from rllab.envs.base import Env
from rllab.spaces import Box, Discrete
from rllab.envs.base import Step
import numpy as np
import gym
import abc
import glob, skimage.io, cv2, os
from natsort import natsorted
from collections import deque

from collections import deque
LEAF_SIZE = 8

class CustomEnv(Env):
    metadata = {'render.modes': ['human']}
    # reward_range = (0.0, 1.0)
    @abc.abstractmethod
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Init CustomEnv")
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment state and returns an initial observation
        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation

    @abc.abstractmethod
    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        pass


class ImageEnv(CustomEnv):
    def __init__(self, dataDir=None, num=10, DIMY=256, DIMX=256):
        # Read the image here             
        self.imageFiles = natsorted (glob.glob(os.path.join(dataDir, 'images/*.*')))[:num]
        self.labelFiles = natsorted (glob.glob(os.path.join(dataDir, 'labels/*.*')))[:num]
        assert len(self.imageFiles) == len(self.labelFiles)
        # print(self.imageFiles)[:1]
        # print(self.labelFiles)[:1]
        self.images = []
        self.labels = []
        for imageFile in self.imageFiles:
            self.images.append( skimage.io.imread(imageFile))
        for labelFile in self.labelFiles:
            self.labels.append( skimage.io.imread(labelFile))       
            
        self.image  = None
        self.label  = None
        self.estim  = None
        self.DIMZ   = None
        self.DIMY   = DIMY
        self.DIMX   = DIMX
        
        
        self.dataDir = dataDir
        self.reset()

    @property
    def reward_range(self):
        return (0.0, 1.0)  

    def reset(self):
        rand_idx = np.random.randint(0, len(self.imageFiles))
        self.image = cv2.imread(self.imageFiles[rand_idx], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        self.label = cv2.imread(self.labelFiles[rand_idx], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        self.estim = np.zeros_like(self.image)

    def step(self, act):
        rwd = 0
        done = False
        info = {'ale.lives' : 0}
        return self.image, rwd, done, info
        
    def render(self, mode='rgb_array'):
        vis = np.concatenate([self.image, self.label], axis=1)
        #print(vis.shape)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        return vis

LEAF_SIZE = 2
class Quad(object):
    def __init__(self, model, box, depth):
        self.model      = model
        self.box        = box
        self.depth      = depth
        
        self.leaf       = self.is_leaf()
        self.area       = self.compute_cent()
        self.cent       = self.compute_area()
        self.children   = []
        self.val        = None

    def is_leaf(self, leaf_size=LEAF_SIZE):
        l, t, r, b = self.box
        return int(r - l <= leaf_size or b - t <= leaf_size)
    
    def is_last(self, leaf_size=LEAF_SIZE):
        l, t, r, b = self.box
        return int(r - l == leaf_size or b - t == leaf_size)
    
    def compute_cent(self):
        l, t, r, b = self.box
        return ((b+t)/2, (l+r)/2)

    def compute_area(self):
        l, t, r, b = self.box
        return (r - l) * (b - t)
    
    def split(self):
        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2
        depth = self.depth + 1
        tl = Quad(self.model, (l, t, lr, tb), depth)
        tr = Quad(self.model, (lr, t, r, tb), depth)
        bl = Quad(self.model, (l, tb, lr, b), depth)
        br = Quad(self.model, (lr, tb, r, b), depth)
        self.children = (tl, tr, bl, br)
        # ct = Quad(self.model, (l+lr/2, t+tb/2, lr+lr/2, tb+tb/2), depth)
        # self.children = (tl, tr, bl, br, ct)
        return self.children

    def get_leaf_nodes(self, max_depth=None):
        if not self.children:
            return [self]
        if max_depth is not None and self.depth >= max_depth:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_nodes(max_depth))
        return result


DIMY = 256
DIMX = 256
class QuadLocEnv(ImageEnv):
    def __init__(self, 
        dataDir=None, 
        num=10, 
        DIMY=256, 
        DIMX=256, 
        env_config=None):

        super(QuadLocEnv, self).__init__(dataDir, num, DIMY, DIMX)
        self.heap = None
        self.root = None 
        
    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        img = self.image.copy()
        est = np.zeros_like(img)

        if self.heap:
            quad = self.heap[-1][-1]
            l, t, r, b = quad.box
            est[t:b, l:r] = 255
        else:
            l, t, r, b = 0, 0, DIMX, DIMY
            est[t:b, l:r] = 255

        self.estim = est
        obs = np.stack((self.image, self.estim), axis=2)
        return obs
       
    def render(self, mode='rgb_array'):
        rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        rgb[...,0] = self.image
        rgb[...,1] = self.label
        rgb[...,2] = self.estim
        return rgb


    @property
    def quads(self):
        return [item for item in self.heap]
    
    def push(self, quad):
        self.heap.append((quad.leaf, quad))
    
    def pop(self, act):
        if self.heap:
            quad =  self.heap.popleft()[-1]
            # print(quad.box)
            quad.val = act
            return quad
        else:
            return None
        
    
    def split(self, act):
        if self.heap:
            # print('ifsplit')
            quad = self.heap[-1][-1]
            if quad.is_last():
                # self.push(quad)
                quad.val = 1 # No need to push it again
            else:
                children = quad.split()
                # for child in children:
                for idx, child in enumerate(children):
                    if idx==act:
                        child.val = act
                        print('Box', child.box)
                        self.push(child)
                    else:
                        pass

           
    def _get_label_center (self):
        if self.label.shape[-1] == 3:
            lbl = cv2.cvtColor(self.label, cv2.COLOR_RGB2GRAY)
        else:
            lbl = self.label
        from skimage.morphology import skeletonize
        ske = skeletonize (lbl == 255)
        index_list = np.where (ske)
        index_mean = (np.mean (index_list[0]), np.mean (index_list[1]))
        index_zip = np.array (zip (index_list[0], index_list[1]))
        centroid_id = np.argmin (np.sum ((index_zip - index_mean) ** 2, axis=1))
        return list(index_zip [centroid_id])

    def _get_agent_center (self):
        quad = self.heap[-1][-1]
        agent_center = list(quad.compute_cent())
        print(agent_center)
        return agent_center

    def step(self, act=None):
        rwd  = 0
        done = False
        info = {'ale.lives' : 0}
        
        quad = self.split(act)

        # Calculate reward
        def l2_distance(x,y):
            return math.sqrt(sum(math.pow(a-b,2) for a, b in zip(x, y)))
        def l1_distance(x,y):
            return sum(abs(a-b) for a, b in zip(x, y))

        rwd = 512-l1_distance(np.array(self._get_label_center()), 
                              np.array(self._get_agent_center()))
        rwd = rwd / 512.0
        if rwd < 0.95: 
            rwd = 0

        if len(self.heap) > 5:
            done = True

        return self.observation_space, rwd, done, info
    

    
    def reset(self):
        super(QuadLocEnv, self).reset()
        self.label[self.label>=128] = 255
        self.label[self.label<128] = 0
        self.image = self.image.astype(np.uint8)
        self.label = self.label.astype(np.uint8)

        self.estim = np.zeros_like(self.image)
        self.heap = deque([])
        self.root = Quad(self, (0, 0, self.DIMX, self.DIMY), 0)
        self.push(self.root)
        # for k in range(21): #85# 4^0 + 4^1 + 4^2 + 4^3 
        #     self.split(act=2)
        return self.observation_space


if __name__ == '__main__':
    env = QuadLocEnv(dataDir='/home/Pearl/quantm/CVPR18/segmentation/data/train/', num=20)
    num = env.action_space.n
    print("Action:", num)


    env.reset()
    isFirst = True
    #for i in range(5):
    while True:
        if isFirst:
            act = int(cv2.waitKey()-ord('0')) #act = 2
            isFirst = False
        else:
            act = int(cv2.waitKey()-ord('0'))
            
        
        obs, rwd, done, info = env.step(act)
        vis = env.render()
        print(act, done)
        cv2.imshow('0', vis[...,0])
        cv2.imshow('1', vis[...,1])
        cv2.imshow('2', vis[...,2])
        cv2.imshow('n', vis)
        # cv2.imshow('3', vis[...,2])
        print("Reward:", rwd)
        if done:
            print("done, press any key to continue!")
            cv2.waitKey()
            env.reset()
        