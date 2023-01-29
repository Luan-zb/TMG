import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from nystrom_attention import NystromAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

#ref link:https://blog.csdn.net/lwf1881/article/details/123673666
class Attention(nn.Module):   # code for Multi-Head Self-Attention
    def __init__(self,
                 dim,             # 输入token的dim
                 num_heads=8,     # the num of heads
                 qkv_bias=False,  # when generate the q  kv,wether to use the bias
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads                        # 计算每一个head的dim，类似均分的感觉
        self.scale = qk_scale or head_dim ** -0.5          # represent for 公式中的1/sqrt(dk)-->即分母
        
        # 这样实现的目的是为了实现并行化
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv是如何生成的，可以通过全连接层来进行转换，别人的代码中会使用三个全连接层来分别进行生成
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)                    # 对拼接后的结果进行相应的映射处理，也是通过全连接层来实现的
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]   ——>num_patches=14×14=196 1：class token   total_embed_dim:768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]   （3：代表q k v三个参数）
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]   （来对数据的顺序进行相应的调整-->为了方便后续来进行相关的运算）
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)   可通过切片的方式来获取qkv的数据信息，每一个的维度信息为（batch_size, num_heads, num_patches + 1, embed_dim_per_head）
        print("wheter q=k",q==k)
        # 此时的所有操作均是针对每一个head来进行操作的。@为矩阵乘法，当是多维数据时，则需要针对最后两个维度来进行矩阵乘法操作。
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        attn = (q @ k.transpose(-2, -1)) * self.scale   ## represent for 公式中的1/sqrt(dk)-->即分母,进行norm处理
        #获得了每个v所对应的权重信息
        attn = attn.softmax(dim=-1)                     # 进行softmax的处理，dim=-1（行处理）其实是针对每一行进行softmax的处理。
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]  （其实是将最后两个维度的信息进行了拼接处理）
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)    #通过Wo矩阵将拼接好的结果进行相应的映射
        x = self.proj_drop(x)  # 通过dropout来获取最终的输出
        return x


#ref:https://zhuanlan.zhihu.com/p/517730496
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, qkv_bias=False, qk_scale=True, attn_drop_ratio=0.,drop_ratio=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        #self.attn = NystromAttention(
        #    dim = dim,
        #    dim_head = dim//8,
        #    heads = 8,
        #    num_landmarks = dim//2,    # number of landmarks
        #    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #    residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #    dropout=0.1
        #)

        self.attn=Attention(dim, num_heads=dim//32, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=dim//32)
    def forward(self, x):
        print("x shape",x.shape)                               # x shape torch.Size([1, 6084, 512])
        print("after norm layer x shape:",self.norm(x).shape)  # after norm layer x shape: torch.Size([1, 6084, 512])
        x = x + self.attn(self.norm(x))
        print("after attn x shape:",x.shape)                   # after attn x shape: torch.Size([1, 6084, 512])
        return x

#LeFF ref link:https://github.com/rishikksh20/CeiT-pytorch/blob/master/module.py
class LeFF(nn.Module):
    def __init__(self,dim = 512, scale = 4, depth_kernel = 3):
        super().__init__()
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] 

        x = self.up_proj(feat_token)
        print("after up_proj x shape:",x.shape)   # after up_proj x shape: torch.Size([1, 2048, 6084])
        B, C, _ = x.shape
        x = x.view(B, C, H, W) 
        print("after view x shape:",x.shape)        # after view x shape: torch.Size([1, 2048, 78, 78])

        x = self.depth_conv(x)
        x = x.flatten(2).transpose(1, 2) 
        print("after flatten and transpose x shape:",x.shape)   # after flatten and transpose x shape: torch.Size([1, 6084, 2048])

        x = self.down_proj(x)
        print("after down_proj x shape:",x.shape)        # after down_proj x shape: torch.Size([1, 6084, 512])
        return x

#https://www.bilibili.com/video/av380166304
class TransformerMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransformerMIL, self).__init__()

        self.pos_layer = LeFF(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
    

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() <1:
            device_ids = list(range(torch.cuda.device_count()))
            self.layer1 = nn.DataParallel(self.layer1, device_ids=device_ids).to('cuda:0')
            self.pos_layer = nn.DataParallel(self.pos_layer, device_ids=device_ids).to('cuda:0')
            self.layer2 = nn.DataParallel(self.layer2, device_ids=device_ids).to('cuda:0')
            self.norm = nn.DataParallel(self.norm, device_ids=device_ids).to('cuda:0')
        else: #DELL--->run 
            self.layer1 = self.layer1.to(device)
            self.pos_layer = self.pos_layer.to(device)
            self.layer2 = self.layer2.to(device)
            self.norm = self.norm.to(device)
        self._fc2 = self._fc2.to(device)

    def forward(self,**kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        print("h shape",h.shape)                    # h shape torch.Size([1, 6000, 1024])
        
        h = self._fc1(h) #[B, n, 512]
        print("after _fc1 layer shape",h.shape)     # after _fc1 layer shape torch.Size([1, 6000, 512])
        #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #h = h.float().to(device) #[B, n, 1024]
        #h = h.expand(1, -1, -1)
        print("h shape",h.shape)                    # h shape torch.Size([1, 6000, 1024])
        
        #h = self._fc1(h) #[B, n, 512]
        #print("after _fc1 layer shape",h.shape)     # after _fc1 layer shape torch.Size([1, 6000, 512])
        
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
        print("after norm layer shape",h.shape)     # after norm layer shape torch.Size([1, 512])

        #---->predict---->类似于MLP Head
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
    model = TransformerMIL(n_classes=5).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)


    #打印的是权重的层的名字和对应形状，顺序可能不是对的
    for ind,i in model.state_dict().items():
        print (ind,i.shape)
    
    for n in (model.modules()):
        print (n)

    for name, module in model._modules.items():
        print (name," : ",module)

    #t = torch.randn(1, 1024, 1024)
    #flops1 = FlopCountAnalysis(model(data=data), data)
    #print("Self-Attention FLOPs:", flops1.total())
