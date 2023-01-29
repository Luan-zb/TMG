import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class MIL_fc(nn.Module):#  n_class=2
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            print("h shape",h.shape)
            logits = self.classifier.module[3](h)
            print("logits shape",logits.shape)
        else:
            logits  = self.classifier(h) # K x 1
            print("logits shape",logits.shape)
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):##  n_class>2
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
    
    def forward(self, h, return_features=False):
        device = h.device
        print("h shape",h.shape)
        h = self.fc(h)
        print("after fc layer shape",h.shape)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)
        print("logits shape",logits.shape)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
                print ("DataParallel Done")
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)   
                print("logits[:, c] shape",logits[:, c].shape)
                print("DataParallel")     

        # print("logits shape",logits.shape)
        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        print("y_probs.view(1, -1) shape",y_probs.view(1, -1).shape)
        print("m shape",m.shape)
        print("m",m)

        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        print("(m // self.n_classes).view(-1, 1)",(m // self.n_classes).view(-1, 1))
        print("(m % self.n_classes).view(-1, 1)",(m % self.n_classes).view(-1, 1))
        print("torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1)",torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1))
        print("top_indices",top_indices)
        

        top_instance = logits[top_indices[0]]
        print("top_indices[0]",top_indices[0])
        print("top_instance",top_instance)

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        print("Y_hat",Y_hat)
        print("Y_prob",Y_prob)
        
        results_dict = {}
        
        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            print("top_features shape",top_features.shape)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = MIL_fc_mc(n_classes=8)
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)

        
