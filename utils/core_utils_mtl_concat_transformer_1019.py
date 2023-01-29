import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from datasets.dataset_mtl_concat import save_splits
from sklearn.metrics import roc_auc_score
#from models.model_toad import TOAD_fc_mtl_concat
from models.TransformerMIL_LeFF_MSA import TransformerMIL
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)  #Saves model when validation loss decrease.
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    #create the "fold" dir in the result
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        '''
        writer = SummaryWriter(log_dir='logs',flush_secs=60)
        log_dir：tensorboard文件的存放路径
        flush_secs：表示写入tensorboard文件的时间间隔
        '''
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets    #datasets = (train_dataset, val_dataset, test_dataset)传入的参数

    #save the train/val/test splits to the  splits_0.csv
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    loss_fn = nn.CrossEntropyLoss()
    
    print('\nInit Model...', end=' ')  #作用：为end传递一个空字符串，这样print函数不会在字符串末尾添加一个换行符，而是添加一个空字符串。
    model_dict = {'n_classes': args.n_classes}

    model= TransformerMIL(**model_dict)#.cuda()
    
    #model = TOAD_fc_mtl_concat(**model_dict)
    
    #model.relocate()  #set the model to GPU
    print('Done!')
    print_network(model)  # print the network structure

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)  #optim:adam  SGD
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    #return either the validation loader or training loader
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    print("hello")
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    """
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 20
        stop_epoch (int): Earliest epoch possible for stopping
        verbose (bool): If True, prints a message for each validation loss improvement. 
                        Default: False
    """
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)
    else:
        early_stopping = None
    print('Done!')

    # start_epoch = -1
    # if args.RESUME:
    #     path_checkpoint = "./models/checkpoint/ckpt_best_284.pth"  # 断点路径
    #     checkpoint = torch.load(path_checkpoint)  # 加载断点
    #     model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    #     optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #     start_epoch = checkpoint['epoch']  # 设置开始的epoch
    # for epoch in range(start_epoch + 1, args.max_epochs):
    #     # def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):
    #     train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
    #     stop = validate(cur, epoch, model, val_loader, args.n_classes,
    #                     early_stopping, writer, loss_fn, args.results_dir)

    #     checkpoint = {
    #         "net": model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         "epoch": epoch
    #     }
    #     if not os.path.isdir("./models/checkpoint"):
    #         os.mkdir("./models/checkpoint")
    #     torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' % (str(epoch)))

    #     if stop:
    #         break

    for epoch in range(args.max_epochs):
        #def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break
    ##获得模型的原始状态以及参数
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, cls_val_error, cls_val_auc, _= summary(model, val_loader, args.n_classes)
    print('Cls Val error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_val_error, cls_val_auc))

    results_dict, cls_test_error, cls_test_auc, acc_loggers= summary(model, test_loader, args.n_classes)
    print('Cls Test error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_test_error, cls_test_auc))
    print(acc_loggers)
    
    #for i in range(args.n_classes):
    #    acc, correct, count = acc_loggers[0].get_summary(i)
    #    print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    #    if writer:
    #        writer.add_scalar('final/test_class_{}_tpr'.format(i), acc, 0)


    if writer:
        writer.add_scalar('final/cls_val_error', cls_val_error, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/cls_test_error', cls_test_error, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)

    
    writer.close()
    return results_dict, cls_test_auc, cls_val_auc, 1-cls_test_error, 1-cls_val_error 


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)

    cls_train_error = 0.
    cls_train_loss = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data =  data.to('cuda:0')
        #print("data shape:",data.shape)
        label = label.to('cuda:1')
        #print("label shape:",label.shape)
        #print("label",label)
    
        results_dict = model(data)#.to('cuda:1')#.to(device)
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        
        cls_logger.log(Y_hat, label)
        
        cls_loss =  loss_fn(logits, label) 
        loss = cls_loss

        cls_loss_value = cls_loss.item()
        cls_train_loss += cls_loss_value

        if (batch_idx + 1) % 5 == 0:
            print('batch {}, cls loss: {:.4f}, '.format(batch_idx, cls_loss_value) + 
                'label: {},  bag_size: {}'.format(label.item(),  data.size(0)))
           
        cls_error = calculate_error(Y_hat, label)
        cls_train_error += cls_error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    cls_train_loss /= len(loader)
    cls_train_error /= len(loader)

    print('Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}'.format(epoch, cls_train_loss, cls_train_error))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_tpr'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/cls_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_error', cls_train_error, epoch)
         
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_val_error = 0.
    cls_val_loss = 0.
    
    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data =  data.to('cuda:0')
            label = label.to('cuda:1')

            results_dict = model(data)
            logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            del results_dict

            cls_logger.log(Y_hat, label)
            
            cls_loss =  loss_fn(logits, label) 
            loss = cls_loss
            cls_loss_value = cls_loss.item()

            cls_probs[batch_idx] = Y_prob.cpu().numpy()
            cls_labels[batch_idx] = label.item()

            
            cls_val_loss += cls_loss_value
            
            cls_error = calculate_error(Y_hat, label)
            cls_val_error += cls_error
            

    cls_val_error /= len(loader)
    cls_val_loss /= len(loader)


    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        cls_aucs = []
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        cls_auc = np.nanmean(np.array(cls_aucs))
    
    
    if writer:
        writer.add_scalar('val/cls_loss', cls_val_loss, epoch)
        writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_error', cls_val_error, epoch)

    print('\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_val_error, cls_auc))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_tpr'.format(i), acc, epoch)
    

    if early_stopping:
        assert results_dir
        early_stopping(epoch, cls_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    site_logger = Accuracy_Logger(n_classes=2)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.

    all_cls_probs = np.zeros((len(loader), n_classes))
    all_cls_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data =  data.to('cuda:0')
        label = label.to('cuda:1')
        #site = site.to(device)
        #sex = sex.float().to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data)

        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        del results_dict

        cls_logger.log(Y_hat, label)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item()}})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error

    cls_test_error /= len(loader)

    if n_classes == 2:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
        
    else:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs, multi_class='ovr')
    

    return patient_results, cls_test_error, cls_auc, (cls_logger)
