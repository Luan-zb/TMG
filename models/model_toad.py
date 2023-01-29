import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np
"""
Attention Network without Gating (2 fc layers) + Tanh()
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
# class Attn_Net(nn.Module):
#     def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
#         super(Attn_Net, self).__init__()
#         self.module = [
#             nn.Linear(L, D),
#             nn.Tanh()]

#         if dropout:
#             self.module.append(nn.Dropout(0.25))

#         self.module.append(nn.Linear(D, n_classes))
        
#         self.module = nn.Sequential(*self.module)
    
#     def forward(self, x):
#         return self.module(x), x # N x n_classes


class Attn_Net(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Linear(D, n_tasks)
        
    
    def forward(self, x):
        A = self.attention_a(x)
        A = self.attention_b(A)  # N x n_classes
        return A, x




"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):  #define the Attn_Net_Gated
    '''
    (6): Attn_Net_Gated(
      (attention_a): Sequential(
        (0): Linear(in_features=512, out_features=384, bias=True)
        (1): Tanh()
        (2): Dropout(p=0.25, inplace=False)
      )
      (attention_b): Sequential(
        (0): Linear(in_features=512, out_features=384, bias=True)
        (1): Sigmoid()
        (2): Dropout(p=0.25, inplace=False)
      )
      (attention_c): Linear(in_features=384, out_features=2, bias=True)
    )
  )
    '''
    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a =[
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout: # whether to add dropout layer
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_tasks)  #represement task number

    def forward(self, x):
        a = self.attention_a(x)

        b = self.attention_b(x)
        #torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
        A = a.mul(b)

        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""
'''
Init Model... Done!
TOAD_fc_mtl_concat(
  (attention_net): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.25, inplace=False)
    (3): Linear(in_features=512, out_features=512, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.25, inplace=False)
    (6): Attn_Net_Gated(
      (attention_a): Sequential(
        (0): Linear(in_features=512, out_features=384, bias=True)
        (1): Tanh()
        (2): Dropout(p=0.25, inplace=False)
      )
      (attention_b): Sequential(
        (0): Linear(in_features=512, out_features=384, bias=True)
        (1): Sigmoid()
        (2): Dropout(p=0.25, inplace=False)
      )
      (attention_c): Linear(in_features=384, out_features=2, bias=True)
    )
  )
  (classifier): Linear(in_features=513, out_features=2, bias=True)
  (site_classifier): Linear(in_features=513, out_features=2, bias=True)
)
Total number of parameters: 1183752
Total number of trainable parameters: 1183752
'''
class TOAD_fc_mtl_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]  #set the size to be deal with
        '''
        extend和append的区别：
        1. 列表可包含任何数据类型的元素，单个列表中的元素无须全为同一类型。
        2.  append() 方法向列表的尾部添加一个新的元素。只接受一个参数。
        3.  extend()方法只接受一个列表作为参数，并将该参数的每个元素都添加到原有的列表中。
        '''
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)  #修改网络结构，改成单任务结构
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(size[1], n_classes)   #(classifier): Linear(in_features=513, out_features=2, bias=True)
        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            '''
            CLASS torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
                module:module to be parallelized
                device_ids:CUDA devices（default all devices）
                output_devices:device location of output(default,devices[0])
                module即表示你定义的模型，device_ids表示你训练的device，output_device这个参数表示输出结果的device.
                最后一个参数output_device一般情况下是省略不写的，那么默认就是在device_ids[0]，也就是第一块卡上，也就解释了为什么第一块卡的显存会占用的比其他卡要更多一些。
            '''
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else: #DELL--->run 
            self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)    #single task:-->A.shap [1350, 1]      h.shape [1350, 512]
        # print("A.shape",A.shape)
        # print("h.shape",h.shape)
        A = torch.transpose(A, 1, 0)  #after A.shape [1, 1350]

        #print("after A.shape",A.shape)
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1)  #按行softmax，行和为1
        # torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
        M = torch.mm(A, h)   #[1, 1350] [1350, 512]
        # print("M.shape",M.shape)       #[1, 512]
        # print("M[0].shape",M[0].shape) #[512])
        # print("M[0].unsqueeze(0) shape",M[0].unsqueeze(0).shape) #[1,512])
        '''
        >>> import torch
        >>> A=torch.ones(2,3)    #2x3的张量（矩阵）                                     
        >>> A
        tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
        >>> D=2*torch.ones(2,4) #2x4的张量（矩阵）
        >>> C=torch.cat((A,D),1)#按维数1（列）拼接
        >>> C
        tensor([[ 1.,  1.,  1.,  2.,  2.,  2.,  2.],
            [ 1.,  1.,  1.,  2.,  2.,  2.,  2.]])
        '''
        #M = torch.cat([M, sex.repeat(M.size(0),1)], dim=1)   #dim=1,代表按列拼接     ->2×513

        #squeeze的用法主要就是对数据的维度进行压缩或者解压
        #self.classifier = nn.Linear(size[1] + 1,n_classes)  # (classifier): Linear(in_features=513, out_features=2, bias=True)
        logits  = self.classifier(M[0].unsqueeze(0))   #M[0].unsqueeze(0):1×512   M[0]:512   1×2

        '''
        torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        沿给定dim维度返回输入张量input中 k 个最大值。
        返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标。
        如果设定布尔值sorted 为_True_，将会确保返回的 k 个值被排序。
        
        参数:
        input (Tensor) – 输入张量
        k (int) – “top-k”中的k
        dim (int, optional) – 排序的维
        largest (bool, optional) – 布尔值，控制返回最大或最小值
        sorted (bool, optional) – 布尔值，控制返回值是否排序
        out (tuple, optional) – 可选输出张量 (Tensor, LongTensor) output buffer
        '''
        Y_hat = torch.topk(logits, 1, dim = 1)[1]  #dim=0表示按照列求topn，dim=1表示按照行求topn  得到的是预测元素下标
        Y_prob = F.softmax(logits, dim = 1)        #按行softmax，行和为1

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})
        '''
        logits：与性别信息进行拼接后的结果，513维特征向量
        Y_prob：多分类任务中的预测概率，和为1
        Y_hat：代表的top-1预测所对应的预测元素的下标
        A:经过attention_net的结果，如果是2个任务：1×2
        '''
        return results_dict

'''
pred = torch.tensor([[-0.5816, -0.3873, -1.0215, -1.0145,  0.4053],
        [ 0.7265,  1.4164,  1.3443,  1.2035,  1.8823],
        [-0.4451,  0.1673,  1.2590, -2.0757,  1.7255],
        [ 0.2021,  0.3041,  0.1383,  0.3849, -1.6311]])
print(pred)
values, indices = pred.topk(1, dim=0, largest=True, sorted=True)
print(indices)
print(values)
# 用max得到的结果，设置keepdim为True，避免降维。因为topk函数返回的index不降维，shape和输入一致。
_, indices_max = pred.max(dim=0, keepdim=True)
print(indices_max)
print(indices_max == indices)
输出：
tensor([[-0.5816, -0.3873, -1.0215, -1.0145,  0.4053],
        [ 0.7265,  1.4164,  1.3443,  1.2035,  1.8823],
        [-0.4451,  0.1673,  1.2590, -2.0757,  1.7255],
        [ 0.2021,  0.3041,  0.1383,  0.3849, -1.6311]])
tensor([[1, 1, 1, 1, 1]])
tensor([[0.7265, 1.4164, 1.3443, 1.2035, 1.8823]])
tensor([[1, 1, 1, 1, 1]])
tensor([[True, True, True, True, True]])
'''
