# -*- coding: utf-8 -*-

import torch
from torch import nn
import cv2
import torchvision.transforms as transforms
from PIL import Image
import pickle

class AutoencoderInference():
    
    def __init__(self,threshold=0.001126351545933871,modelName='./utils/model_name',resize=(480,240) ,crop=(120,240,0,480),tensorResize=(128,512),cuda=True):
        self.rWidth,self.rHeight = resize
        self.cropRange = crop
        self.scaler = transforms.Resize(tensorResize)
        self.to_tensor = transforms.ToTensor()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = torch.load((modelName if cuda else modelName+'-cpu')+'.pkl').eval()
        self.loss_func = nn.MSELoss(reduction='none')
        self.threshold = threshold
    
    def image_resize(self,image, inter = cv2.INTER_AREA, pad_value=(0, 0, 0)):
        
        dim = None 
        (h, w) = image.shape[:2]
        r_h = self.rHeight/float(h)
        r_w = self.rWidth/float(w)
    
        if(r_h<r_w):
            dim = (int(w * r_h), self.rHeight) 
        elif(r_h>r_w):
            dim = (self.rWidth, int(h * r_w)) 
        else:
            dim = (self.rWidth, self.rHeight)
        
        pad_w = int((self.rWidth - dim[0]) /2)
        pad_h = int((self.rHeight - dim[1]) /2)
    
        resized = cv2.resize(image, dim, interpolation = inter) 
        pad_img = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    
        return pad_img
    
    def cropConvert(self,image):
        image = self.image_resize(image)
        image = image[self.cropRange[0]:self.cropRange[1],self.cropRange[2]:self.cropRange[3]]
        #cv2.imwrite('temp.jpg',image)
        #image = Image.open('temp.jpg').convert('RGB')
        #image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        
        image = Image.fromarray(image)
        return image
    
    def batchNorm(self,img):                   
        a = [self.to_tensor(self.scaler(self.cropConvert(im))) for im in img]
        images = torch.stack(a).to(self.device)
        return images
    
    def inference(self,image,Batch=False):
        if not Batch:
            image = [image]
        batch = self.batchNorm(image)
        reconstruct = self.model(batch)
        loss = self.loss_func(reconstruct,batch)
        result = [l.mean() > self.threshold for l in loss.data.cpu().numpy()]
        if not Batch:
            result = result[0]
        return result
        #return result,float(loss.mean())


if __name__ == '__main__':
    AE = AutoencoderInference(cuda = False)
    img  = cv2.imread('./utils/test.jpg')
    result = AE.inference(img)
    print(result)