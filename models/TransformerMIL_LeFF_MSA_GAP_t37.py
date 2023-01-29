import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from performer_pytorch import Performer
from performer_pytorch import SelfAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

#ref:https://zhuanlan.zhihu.com/p/517730496
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)

        #Attention (4):Performer with SelfAttention##-->ref link:https://github.com/lucidrains/performer-pytorch.without dropout
        self.attn = SelfAttention(
            dim = 512,
            heads = 8,
            causal = False
        )  
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        #print("after attn x shape:",x.shape)                   # after attn x shape: torch.Size([1, 6084, 512])
        return x





#test 9:LeFF(cnn_feat+proj+proj1+proj2)+Performer+MIL,the LeFF(cnn_feat+proj+proj1+proj2)-->PPEG,##Attention (4):Performer with SelfAttention##
class LeFF(nn.Module):
   def __init__(self, dim=512):
       super(LeFF, self).__init__()
       # group convolution ref link:https://www.jianshu.com/p/a936b7bc54e3
       self.proj = nn.Conv2d(dim, dim, 1, 1, 1//2, groups=dim)  
       self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
       self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)


   def forward(self, x, H, W):
       B, _, C = x.shape
       #print("x.shape:",x.shape)                    # x.shape torch.Size([1, 6085, 512])

       cls_token, feat_token = x[:, 0], x[:, 1:]   # split x into cls_token-->torch.Size([1, 512]) , feat_token-->torch.Size([1, 6084, 512]) 

       cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)  # 1：patch_size  2:patch_dim
       #print("after transpose feat_token shape:", feat_token.transpose(1, 2).shape)  #  torch.Size([1, 512, 6084])
       #print("cnn_feat shape:",cnn_feat.shape)      # cnn_feat shape torch.Size([1, 512, 78, 78])

       x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)#+self.proj3(cnn_feat)                           #cnn_feat add to the (obtained from convolution block processing with kernal k=3,5,7 padding=1,2,3)
       #print("x shape:",x.shape)                    # x shape: torch.Size([1, 512, 78, 78])

       x = x.flatten(2).transpose(1, 2)             
       #print("after flatten x shape:",x.shape)      # after flatten x shape: torch.Size([1, 6084, 512])

       x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  
       #print("after concat x shape:",x.shape)       # after concat x shape: torch.Size([1, 6085, 512])
       return x


#https://www.bilibili.com/video/av380166304
class TransformerMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransformerMIL, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.pos_layer1 = LeFF(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.pos_layer2 = LeFF(dim=512)
        #add another layer
        self.layer3 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(8)
        self._fc2 = nn.Linear(8, self.n_classes)

        self.conv1=nn.Conv1d(512,8,1)
    

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() <1:
            device_ids = list(range(torch.cuda.device_count()))
            self.layer1 = nn.DataParallel(self.layer1, device_ids=device_ids).to('cuda:0')
            self.pos_layer1 = nn.DataParallel(self.pos_layer, device_ids=device_ids).to('cuda:0')
            self.layer2 = nn.DataParallel(self.layer2, device_ids=device_ids).to('cuda:0')
            self.pos_layer2 = nn.DataParallel(self.pos_layer, device_ids=device_ids).to('cuda:0')
            self.norm = nn.DataParallel(self.norm, device_ids=device_ids).to('cuda:0')
        else: #DELL--->run 
            self.layer1 = self.layer1.to(device)
            self.pos_layer1 = self.pos_layer1.to(device)
            self.layer2 = self.layer2.to(device)
            self.norm = self.norm.to(device)
        self._fc2 = self._fc2.to(device)


    # def forward(self, **kwargs):
    #     h = kwargs['data'].float() #[B, n, 1024]
    #     print("original h shape:",h.shape)


    def forward(self,h):
        h = h.float()#.to(device) #[B, n, 1024]
        h = h.expand(1, -1, -1)
        
        h = self._fc1(h) #[B, n, 512]
        # print("after _fc1 layer shape",h.shape)     # after _fc1 layer shape torch.Size([1, 6000, 512])
        
        #---->pad
        H = h.shape[1]
        # print("H shape",H)                          # H shape 6000
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # print("_H,_W",_H,_W)                        # _H,_W 78 78
        add_length = _H * _W - H
        # print("add_length",add_length)              # add_length 84


        #add the code to deal with add_length
        #feature token concat the first add_length feature token
        if add_length!=0:
            h = torch.cat([h, h[:,-add_length:,:]],dim = 1) #[B, N, 512]
            # print("----h shape----",h.shape)

        #print("h.shape",h.shape)                    # h.shape torch.Size([1, 6084, 512])

        #---->cls_token
        B = h.shape[0]
        # print("B",B)                                # B 1
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print("cls_tokens shape",cls_tokens.shape)  # cls_tokens shape torch.Size([1, 1, 512])
        h = torch.cat((cls_tokens, h), dim=1)
        # print("h shape",h.shape)                    # h shape torch.Size([1, 6085, 512])

        #---->Translayer x1------->MSA(h)
        h = self.layer1(h) #[B, N, 512]
        # print("after layer1 h shape",h.shape)       # after layer1 h shape torch.Size([1, 6085, 512])

        #---->LeFF(Locally Enhanced Feed-Forward)
        # Linear Projection-->Spatial Restoration-->Depth-wise Convolution-->Flatten
        h = self.pos_layer1(h, _H, _W) #[B, N, 512]
        # print("after pos_layer h shape",h.shape)    # after pos_layer h shape torch.Size([1, 6085, 512])

        #---->Translayer x2------->MSA(h)
        h = self.layer2(h) #[B, N, 512]
        # print("after layer2 shape",h.shape)         # after layer2 shape torch.Size([1, 6085, 512]

        # ---->LeFF(Locally Enhanced Feed-Forward)
        h = self.pos_layer2(h, _H, _W) #[B, N, 512]


        #update the GAP
        h = self.layer3(h) #[B, N, 512]


        h=h[:,1:].expand(1, -1, -1).transpose(1, 2)
        h=self.conv1(h)
        # print("after con1 shape:",h.shape)                     # torch.Size([1, 8, 6084])

        B, C , _= h.shape
        h=h.view(B, C, _H, _W)   
        # print("after view h shape:",h.shape)                             #after view h shape: torch.Size([1, 8, 78, 78])
        h=torch.nn.functional.adaptive_avg_pool2d(h, (1,1))             
        # print("after adaptive_avg_pool2d h shape:",h.shape)              # after adaptive_avg_pool2d h shape: torch.Size([1, 8, 1, 1])
        h = h.flatten(2).transpose(1, 2)                                 #  torch.Size([1, 1, 8])
        # print("after flatten  shape:",h.shape)
        # print("after flatten transpose h shape:",h.shape)

        # print("class_token2 shape:",h.shape)             #class_token2 shape: torch.Size([1, 512])

        #update the GAP
        h = self.norm(h)[:,0]
        # print("after norm layer shape:",self.norm(h).shape)
        # print("after norm layer shape",h.shape)     # after norm layer shape torch.Size([1, 512])

        #---->predict---->类似于MLP Head
        # logits = self._fc2(h) #[B, n_classes]
        # print("after logits shape",logits.shape)    # after _f2 shape torch.Size([1, 2])

        Y_hat = torch.argmax(h, dim=1)         # original the predicted result
        # print("Y_hat shape",Y_hat.shape)            # Y_hat shape torch.Size([1])
        
        Y_prob = F.softmax(h, dim = 1)         # after softmax:--->predicted result
        # print("Y_prob shape",Y_prob.shape)          # Y_prob shape torch.Size([1, 2])

        results_dict = {'logits': h, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        #print(results_dict)                         # {'logits': tensor([[0.4693, 0.3050]], device='cuda:0', grad_fn=<AddmmBackward0>), 'Y_prob': tensor([[0.5410, 0.4590]], device='cuda:0', grad_fn=<SoftmaxBackward0>), 'Y_hat': tensor([0], device='cuda:0')}
        return results_dict
        
if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = TransformerMIL(n_classes=8)
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
