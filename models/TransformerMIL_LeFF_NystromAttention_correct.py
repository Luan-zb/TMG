import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#ref:https://zhuanlan.zhihu.com/p/517730496
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
    def forward(self, x):
        print("x shape",x.shape)                               # x shape torch.Size([1, 6084, 512])
        print("after norm layer x shape:",self.norm(x).shape)  # after norm layer x shape: torch.Size([1, 6084, 512])
        x = x + self.attn(self.norm(x))
        print("after attn x shape:",x.shape)                   # after attn x shape: torch.Size([1, 6084, 512])
        return x

##LeFF ref link:https://github.com/rishikksh20/CeiT-pytorch/blob/master/module.py
#class LeFF(nn.Module):
#    def __init__(self,dim = 512, scale = 4, depth_kernel = 3):
#        super().__init__()
#        scale_dim = dim*scale
#        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
#                                    Rearrange('b n c -> b c n'),
#                                    nn.BatchNorm1d(scale_dim),
#                                    nn.GELU(),
#                                    )
        
#        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
#                          nn.BatchNorm2d(scale_dim),
#                          nn.GELU(),
#                          )
        
#        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
#                                    Rearrange('b n c -> b c n'),
#                                    nn.BatchNorm1d(dim),
#                                    nn.GELU(),
#                                    Rearrange('b c n -> b n c')
#                                    )
#    def forward(self, x, H, W):
#        B, _, C = x.shape
#        cls_token, feat_token = x[:, 0], x[:, 1:] 

#        x = self.up_proj(feat_token)
#        print("after up_proj x shape:",x.shape)   # after up_proj x shape: torch.Size([1, 2048, 6084])
#        B, C, _ = x.shape
#        x = x.view(B, C, H, W) 
#        print("after view x shape:",x.shape)        # after view x shape: torch.Size([1, 2048, 78, 78])

#        x = self.depth_conv(x)
#        x = x.flatten(2).transpose(1, 2) 
#        print("after flatten and transpose x shape:",x.shape)   # after flatten and transpose x shape: torch.Size([1, 6084, 2048])

#        x = self.down_proj(x)
#        print("after down_proj x shape:",x.shape)        # after down_proj x shape: torch.Size([1, 6084, 512])
#        return x

class LeFF(nn.Module):
    def __init__(self, dim=512):
        super(LeFF, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        
        self.pos_layer = LeFF(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        print("h shape",h.shape)                    # h shape torch.Size([1, 6000, 1024])
        
        h = self._fc1(h) #[B, n, 512]
        print("after _fc1 layer shape",h.shape)     # after _fc1 layer shape torch.Size([1, 6000, 512])
        
        #---->pad
        H = h.shape[1]
        print("H shape",H)                          # H shape 6000
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        print("_H,_W",_H,_W)                        # _H,_W 78 78
        add_length = _H * _W - H
        print("add_length",add_length)              # add_length 84
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        print("h.shape",h.shape)                    # h.shape torch.Size([1, 6084, 512])

        #---->cls_token
        B = h.shape[0]
        print("B",B)                                # B 1
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda() 
        print("cls_tokens shape",cls_tokens.shape)  # cls_tokens shape torch.Size([1, 1, 512])
        h = torch.cat((cls_tokens, h), dim=1)
        print("h shape",h.shape)                    # h shape torch.Size([1, 6085, 512])

        #---->Translayer x1------->MSA(h)
        h = self.layer1(h) #[B, N, 512]
        print("after layer1 h shape",h.shape)       # after layer1 h shape torch.Size([1, 6085, 512])

        #---->LeFF(Locally Enhanced Feed-Forward)
        # Linear Projection-->Spatial Restoration-->Depth-wise Convolution-->Flatten
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        print("after pos_layer h shape",h.shape)    # after pos_layer h shape torch.Size([1, 6085, 512])
        
        #---->Translayer x2------->MSA(h)
        h = self.layer2(h) #[B, N, 512]
        print("after layer2 shape",h.shape)         # after layer2 shape torch.Size([1, 6085, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        print("after norm layer shape",self.norm(h).shape)     # after norm layer shape torch.Size([1, 512])

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        print("after logits shape",logits.shape)    # after _f2 shape torch.Size([1, 2])

        Y_hat = torch.argmax(logits, dim=1)         # original the predicted result
        print("Y_hat shape",Y_hat.shape)            # Y_hat shape torch.Size([1])
        
        Y_prob = F.softmax(logits, dim = 1)         # after softmax:--->predicted result
        print("Y_prob shape",Y_prob.shape)          # Y_prob shape torch.Size([1, 2])

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        print(results_dict)                         # {'logits': tensor([[0.4693, 0.3050]], device='cuda:0', grad_fn=<AddmmBackward0>), 'Y_prob': tensor([[0.5410, 0.4590]], device='cuda:0', grad_fn=<SoftmaxBackward0>), 'Y_hat': tensor([0], device='cuda:0')}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=8).cuda()
    print("model.eval()",model.eval())
    results_dict = model(data = data)
    print(results_dict)
