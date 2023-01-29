import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from performer_pytorch import SelfAttention

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

#ref:https://zhuanlan.zhihu.com/p/517730496
class TransLayer(nn.Module):
    def __init__(self, depth,norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)

        self.attn = SelfAttention(
            dim = 512,
            heads = 8,
            causal = False
        )
        # self.FFL=FeedForward(dim,dim*4)

        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim)),
                PreNorm(dim, FeedForward(dim, dim*4))
        ]))

    def forward(self, x):
        for attn,ff in self.layers:
             x=x+attn(x)
             x=ff(x)+x
        return x

#test 8:LeFF(cnn_feat+proj+proj1+proj2)+NystromAttention+MIL,the LeFF(cnn_feat+proj+proj1+proj2)-->PPEG,##Attention (4):Performer with SelfAttention##,with dropout=0.1
class LeFF(nn.Module):
   def __init__(self, dim=512):
       super(LeFF, self).__init__()
       #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,groups)
       # group convolution ref link:https://www.jianshu.com/p/a936b7bc54e3
       self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)  
       # 在处理的时候是按照every patch dim进行分组处理，每个组的size【_H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))】，通过卷积核大小为7×7，stride=1，padding=3，可以得到与原来大小相同的feature map
       self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
       self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

   def forward(self, x, H, W):
       #print("-----------PPEG-----------")
       B, _, C = x.shape

       cls_token, feat_token = x[:, 0], x[:, 1:]   # split x into cls_token-->torch.Size([1, 512]) , feat_token-->torch.Size([1, 6084, 512]) 

       cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)  # 1：patch_size  2:patch_dim

       x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)                     #cnn_feat add to the (obtained from convolution block processing with kernal k=3,5,7 padding=1,2,3)
       x = x.flatten(2).transpose(1, 2)             # 从第二个维度打平后，再交换成原来的维度。
       x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  #拼接上原来的class_tokon
       return x

#https://www.bilibili.com/video/av380166304
class TransformerMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransformerMIL, self).__init__()

        self.pos_layer = LeFF(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        # self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512,depth=12)
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
            # self.layer1 = self.layer1.to(device)
            self.pos_layer = self.pos_layer.to(device)
            self.layer2 = self.layer2.to(device)
            self.norm = self.norm.to(device)
        self._fc2 = self._fc2.to(device)


    # def forward(self,h):
    #     h = h.float()#.to(device) #[B, n, 1024]
    #     h = h.expand(1, -1, -1)
    #     h = self._fc1(h) #[B, n, 512]

    #test in the single model,without training
    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        print("original h shape:",h.shape)
        h = self._fc1(h) #[B, n, 512]

        #---->pad
        H = h.shape[1]
        #print("H shape",H)                          # H shape 6000
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
   
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda() 
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1------->MSA(h)
        # h = self.layer1(h) #[B, N, 512]

        #---->LeFF(Locally Enhanced Feed-Forward)
        # Linear Projection-->Spatial Restoration-->Depth-wise Convolution-->Flatten
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2------->MSA(h)
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict---->类似于MLP Head
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)         # original the predicted result
        Y_prob = F.softmax(logits, dim = 1)         # after softmax:--->predicted result
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        #print(results_dict)                         # {'logits': tensor([[0.4693, 0.3050]], device='cuda:0', grad_fn=<AddmmBackward0>), 'Y_prob': tensor([[0.5410, 0.4590]], device='cuda:0', grad_fn=<SoftmaxBackward0>), 'Y_hat': tensor([0], device='cuda:0')}
        return results_dict


        
if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransformerMIL(n_classes=5).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)