
import  torch
import  os, glob
import csv
from Compose import *

from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import random
from load_csv_class import id_target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob,R):
        self.flip_prob = flip_prob
        self.R = R

    def __call__(self, image):
        if self.R< self.flip_prob:
            image = F.hflip(image)
        else:
            image=image
        return image
class RandomRotation(object):
    def __init__(self, flip_prob,R):
        self.flip_prob = flip_prob
        self.R=R

    def __call__(self, image):
        if self.R < self.flip_prob:
            image = F.rotate(image,15)
        else:
            image=image
        return image

class mydata(Dataset):
    def __init__(self,root,resize,mode):
        super(mydata, self).__init__
        self.root = root
        self.resize = resize
        # self.name2label = {}
        # root='D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets'
        self.images_path = os.path.join(self.root, './image')
        self.masks_path=os.path.join(self.root, './mask')

        # print(self.name2label)
        self.images,self.masks,self.target = self.load_csv('image.csv')
        if mode=='train':
            self.images = self.images[:int(len(self.images)*0.6)]
            self.masks=self.masks[:int(len(self.masks)*0.6)]
            self.target = self.target[:int(len(self.target) * 0.6)]
        elif mode=='test':
            self.images = self.images[int(len(self.images) * 0.6):int(len(self.images) * 0.8)]
            self.masks = self.masks[int(len(self.masks) * 0.6):int(len(self.masks) * 0.8)]
            self.target= self.target[int(len(self.target) * 0.6):int(len(self.target) * 0.8)]
        else:
            self.images = self.images[int(len(self.images) * 0.8):]
            self.masks = self.masks[int(len(self.masks) * 0.8):]
            self.target = self.target[int(len(self.target) * 0.8):]


    def load_csv(self,filename):
        images=[]
        masks=[]
        #'trainset\\leibie\\00001.jpg'

        images+=glob.glob(os.path.join(self.images_path,'*.png'))
        masks+=glob.glob(os.path.join(self.masks_path,'*.png'))

        np.random.seed(2021)
        images = np.array(images)
        masks = np.array(masks)

        if  not os.path.exists(os.path.join(self.root,filename)):
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer=csv.writer(f)
                ids,targets=id_target()

                for img,mask in zip(images,masks):
                    id= img.split('\\')[-1].split('.')[0]

                    target=targets[id]

                    writer.writerow([img,mask,target])
                print("written into csv file",filename)
        images1 = []
        masks1 = []
        target1=[]
        with open(os.path.join(self.root,filename)) as f:
            reader=csv.reader(f)
            for row in reader:
                img,mask,target=row

                images1.append(img)
                masks1.append(mask)
                target1.append(target)



        assert len(images1) ==len(masks1)==len(target1)
        return images1,masks1,target1


    def __len__(self):
        return len(self.images)
    # def denormalize(self,x_hat):
    #     mean=[0.485,0.456,0.406]
    #     std=[0.229,0.224,0.225]
    #     mean=torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    #     std=torch.tensor(std).unsqueeze(1).unsqueeze(1)
    #     x=x_hat*std+mean
    #     return x

    def __getitem__(self, idx):
        img, mask ,tgt= self.images[idx], self.masks[idx], self.target[idx]
        R = random.uniform(0,1)
        tf1=transforms.Compose([

            lambda x:Image.open(x).convert('RGB'),
            # lambda x: Image.open(x),
            transforms.Resize((self.resize,self.resize)),
            RandomHorizontalFlip(0.5,R),
            RandomRotation(0.5,R),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.13438404, 0.13438404, 0.13438404], [0.16724404, 0.16724404, 0.16724404])

        ])
        tf2 = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((self.resize, self.resize)),
            RandomHorizontalFlip(0.5, R),
            RandomRotation(0.5, R),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),

        ])


        img=tf1(img)
        mask=tf2(mask)
        mask =torch.squeeze(mask).type(torch.long)
        # tgt=torch.from_numpy(tgt)
        return img,mask
