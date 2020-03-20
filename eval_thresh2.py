from __future__ import print_function
from __future__ import division
import pickle
import torch
import os
import sys
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter  # for pytorch below 1.14
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14
from cifar10 import CIFAR100
import matplotlib.pyplot as plt

classes_c2f_map = [['beaver','dolphin','otter','seal','whale'],
    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    ['orchid','poppy','rose','sunflower','tulip'],
    ['bottle','bowl','can','cup','plate'],
    ['apple','mushroom','orange','pear','sweet_pepper'],
    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    ['bed','chair','couch','table','wardrobe'],
    ['bee','beetle','butterfly','caterpillar','cockroach'],
    ['bear','leopard','lion','tiger','wolf'],
    ['bridge','castle','house','road','skyscraper'],
    ['cloud','forest','mountain','plain','sea'],
    ['camel','cattle','chimpanzee','elephant','kangaroo'],
    ['fox','porcupine','possum','raccoon','skunk'],
    ['crab','lobster','snail','spider','worm'],
    ['baby','boy','girl','man','woman'],
    ['crocodile','dinosaur','lizard','snake','turtle'],
    ['hamster','mouse','rabbit','shrew','squirrel'],
    ['maple_tree','oak_tree','palm_tree','pine_tree','willow_tree'],
    ['bicycle','bus','motorcycle','pickup_truck','train'],
    ['lawn_mower','rocket','streetcar','tank','tractor']]
mmap = None
fine = None
coarse = None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # auxiliary
        self.dropout_conv = nn.Dropout(p=0.15)
        self.dropout_fc = nn.Dropout(p=0.4)
        # block 1
        self.conv_b1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_b1_bn1 = nn.BatchNorm2d(32)
        self.conv_b1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_b1_bn2 = nn.BatchNorm2d(32)
        self.conv_b1_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_b1_bn3 = nn.BatchNorm2d(32)
        self.pool_b1 = nn.MaxPool2d(2,2)
        # block 2
        self.conv_b2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_b2_bn1 = nn.BatchNorm2d(64)
        self.conv_b2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_b2_bn2 = nn.BatchNorm2d(64)
        self.conv_b2_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_b2_bn3 = nn.BatchNorm2d(64)
        self.pool_b2 = nn.MaxPool2d(2,2)
        # block 3
        self.conv_b3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_b3_bn1 = nn.BatchNorm2d(128)
        self.conv_b3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_b3_bn2 = nn.BatchNorm2d(128)
        self.conv_b3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_b3_bn3 = nn.BatchNorm2d(128)
        self.pool_b3 = nn.MaxPool2d(2,2)
        # block 4
        self.conv_b4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv_b4_bn1 = nn.BatchNorm2d(256)
        self.conv_b4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_b4_bn2 = nn.BatchNorm2d(256)
        self.conv_b4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_b4_bn3 = nn.BatchNorm2d(256)
        self.pool_b4 = nn.MaxPool2d(2,2)
        # block 5
        self.conv_b5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_b5_bn1 = nn.BatchNorm2d(512)
        self.conv_b5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_b5_bn2 = nn.BatchNorm2d(512)
        self.conv_b5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_b5_bn3 = nn.BatchNorm2d(512)
        self.pool_b5 = nn.MaxPool2d(2,2)

        # coarse branch 1 -> generalized classes
        self.bra1_fc1 = nn.Linear(512, 4096)
        self.bra1_bn1 = nn.BatchNorm1d(4096)
        self.bra1_fc2 = nn.Linear(4096,4096)
        self.bra1_bn2 = nn.BatchNorm1d(4096)
        self.bra1_fc3 = nn.Linear(4096,20)
        self.bra1_sm = nn.Softmax(dim=1)
        # branch 2 -> Fine layers
        self.bra2_fc1 = nn.Linear(512, 4096)
        self.bra2_bn1 = nn.BatchNorm1d(4096)
        self.bra2_fc2 = nn.Linear(4096,4096)
        self.bra2_bn2 = nn.BatchNorm1d(4096)
        self.bra2_fc3 = nn.Linear(4096,100)
        self.bra2_sm = nn.Softmax(dim=1)


    def forward(self, x):
        # block 1 forward prop
        x = F.leaky_relu(self.conv_b1_bn1(self.conv_b1_1(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b1_bn2(self.conv_b1_2(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b1_bn3(self.conv_b1_3(x)))
        x = self.dropout_conv(x)
        x = self.pool_b1(x)
        # block 2 forward prop
        x = F.leaky_relu(self.conv_b2_bn1(self.conv_b2_1(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b2_bn2(self.conv_b2_2(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b2_bn3(self.conv_b2_3(x)))
        x = self.dropout_conv(x)
        x = self.pool_b2(x)
        # block 3 forward prop
        x = F.leaky_relu(self.conv_b3_bn1(self.conv_b3_1(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b3_bn2(self.conv_b3_2(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b3_bn3(self.conv_b3_3(x)))
        x = self.dropout_conv(x)
        x = self.pool_b3(x)
        # block 3 forward prop
        x = F.leaky_relu(self.conv_b4_bn1(self.conv_b4_1(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b4_bn2(self.conv_b4_2(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b4_bn3(self.conv_b4_3(x)))
        x = self.dropout_conv(x)
        x = self.pool_b4(x)
        # block 3 forward prop
        x = F.leaky_relu(self.conv_b5_bn1(self.conv_b5_1(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b5_bn2(self.conv_b5_2(x)))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b5_bn3(self.conv_b5_3(x)))
        x = self.dropout_conv(x)
        x = self.pool_b5(x)
        # offshoot branch 1
        bra1_x = x.view(-1, self.num_flat_features(x))
        bra1_x = self.bra1_fc1(bra1_x)
        bra1_x = self.bra1_bn1(bra1_x)
        bra1_x = F.leaky_relu(bra1_x)
        bra1_x = self.dropout_fc(bra1_x) 
        bra1_x = self.bra1_fc2(bra1_x)
        bra1_x = self.bra1_bn2(bra1_x)
        bra1_x = F.leaky_relu(bra1_x)
        bra1_x = self.dropout_fc(bra1_x) 
        bra1_x = self.bra1_fc3(bra1_x)
        bra1_x = self.bra1_sm(bra1_x)
        # offshoot branch 2
        bra2_x = x.view(-1, self.num_flat_features(x))
        bra2_x = self.bra2_fc1(bra2_x)
        bra2_x = self.bra2_bn1(bra2_x)
        bra2_x = F.leaky_relu(bra2_x)
        bra2_x = self.dropout_fc(bra2_x) 
        bra2_x = self.bra2_fc2(bra2_x)
        bra2_x = self.bra2_bn2(bra2_x)
        bra2_x = F.leaky_relu(bra2_x)
        bra2_x = self.dropout_fc(bra2_x) 
        bra2_x = self.bra2_fc3(bra2_x)
        bra2_x = self.bra2_sm(bra2_x)
        return(bra1_x,bra2_x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def map_subclasses(array):
    for i in range(0,len(array)):
        array[i] = mmap[array[i]]
    return(array)



def eval_net(dataloader):
    correct_coarse = 0
    correct_fine = 0
    total = 0
    total_loss_coarse = 0
    total_loss_fine = 0
    net.eval() # Why would I do this?
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        coarse_labels = np.copy(labels.numpy())
        coarse_labels = map_subclasses(coarse_labels)
        coarse_labels = torch.from_numpy(coarse_labels)
        images, labels, coarse_labels = Variable(images).cuda(), Variable(labels).cuda(), Variable(coarse_labels).cuda()
        coarse_out, fine_out = net(images)
        _, predicted_coarse = torch.max(coarse_out.data, 1)
        _, predicted_fine = torch.max(fine_out.data, 1)
        total += labels.size(0)
        correct_coarse += (predicted_coarse == coarse_labels.data).sum()
        correct_fine += (predicted_fine == labels.data).sum()
        fine_loss = criterion1(fine_out, labels)
        coarse_loss = criterion2(coarse_out, coarse_labels)
        total_loss_fine += fine_loss.item()
        total_loss_coarse += coarse_loss.item()
    net.train() # Why would I do this?
    return (total_loss_coarse / total, correct_coarse.float() / total,
        total_loss_fine/total, correct_fine.float()/ total)

def eval_net3(dataloader,plotter,alpha):
    correct_coarse = 0
    correct_coarse2 = 0
    correct_coarse3 = 0
    correct_fine = 0
    correct_fine2 = 0
    correct_fine3 = 0
    unk_coarse = 0
    total = 0
    total_loss_coarse = 0
    total_loss_fine = 0
    net.eval() # Why would I do this?
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        coarse_labels = np.copy(labels.numpy())
        coarse_labels = map_subclasses(coarse_labels)
        coarse_labels = torch.from_numpy(coarse_labels)
        images, labels, coarse_labels = Variable(images).cuda(), Variable(labels).cuda(), Variable(coarse_labels).cuda()
        #myim = np.transpose(images[12].cpu(),(1,2,0))
        #myim = myim/2 + 0.5
        #plt.imshow(myim)
        #plt.show()
        #sys.exit(1)
        coarse_out, fine_out = net(images)
        top_prob_c, top_label_c = torch.topk(coarse_out, 20)
        top_prob_f, top_label_f = torch.topk(fine_out, 20)
        #print(top_prob_c, top_label_c)
        _, predicted_coarse = torch.max(coarse_out.data, 1)
        _, predicted_fine = torch.max(fine_out.data, 1)
        total += labels.size(0)
        top_label_c1 = top_label_c[:,0]
        top_prob_c1 = top_prob_c[:,0]
        top_prob_c2 = top_prob_c[:,1]
        top_prob_c3 = top_prob_c[:,2]
        top_prob_f1 = top_prob_f[:,0]
        #print(top_prob_c1, top_prob_f1)
        #sys.exit(1)
        top_label_c2 = top_label_c[:,1]
        top_label_c3 = top_label_c[:,2]
        top_label_f1 = top_label_f[:,0]
        top_label_f2 = top_label_f[:,1]
        top_label_f3 = top_label_f[:,2]
            
        correct_coarse += (top_label_c1 == coarse_labels.data).sum()
        correct_coarse2 = correct_coarse2 + (top_label_c2 == coarse_labels.data).sum()
        correct_coarse3 = correct_coarse3 + (top_label_c3 == coarse_labels.data).sum()
        check1 = (top_label_c1 == coarse_labels.data)
        top_fine = top_prob_f[:,0]
        
        num_iter = list(top_fine.size())
        for i in range(0,num_iter[0]):
            for j in range(0,20):
                b = top_label_c[i,0].cpu().numpy()
                a = top_label_f[i,j].cpu().numpy()
                a = mmap[a]
         #       print(i, a, b)
                if(b == a):
                    break
            top_fine[i] = top_prob_f[i,j]
        '''
        for i in range(0,len(top_fine)):
            print(top_fine[i]
        '''
        #divider = (top_prob_f1 / top_prob_c1)
        divider = (top_fine / top_prob_c1)
        check3 = (labels == top_label_f1)
        check2 = (divider < alpha) 
        check1 = check1.int()
        check2 = check2.int()
        check1[check1 == 0] = -1
        check3 = check3.int()
        check3[check3 == 0] = -2
        check4 = (check3 == check1)
        check4[check4 == 0] = -1
        unk_coarse += (check4 == check2).sum() 
        #unk_coarse += ((top_label_c1 == coarse_labels.data) and (top_prob_f1*2 < top_prob_c1)).sum() 
        #unk_coarse += (top_label_c1 == coarse_labels.data)and(top_prob_f1 < (top_prob_c1 - (alpha * top_prob_c1))).sum() 
        correct_fine += (top_label_f1 == labels.data).sum()
        correct_fine2 = correct_fine2 + (top_label_f2 == labels.data).sum()
        correct_fine3 = correct_fine3 + (top_label_f3 == labels.data).sum()
        fine_loss = criterion1(fine_out, labels)
        coarse_loss = criterion2(coarse_out, coarse_labels)
        total_loss_fine += fine_loss.item()
        total_loss_coarse += coarse_loss.item()
    net.train() # Why would I do this?
    correct_coarse2 += correct_coarse
    correct_coarse3 += correct_coarse2
    correct_fine2 += correct_fine
    correct_fine3 += correct_fine2
    return (correct_coarse.float() / total, correct_fine.float()/ total, unk_coarse.float()/total,
            correct_coarse2.float() / total, correct_fine2.float()/ total,
            correct_coarse3.float() / total, correct_fine3.float()/ total)

def eval_net2(dataloader,plotter,alpha):
    correct_coarse = 0
    correct_coarse2 = 0
    correct_coarse3 = 0
    correct_fine = 0
    correct_fine2 = 0
    correct_fine3 = 0
    unk_coarse = 0
    total = 0
    total_loss_coarse = 0
    total_loss_fine = 0
    net.eval() # Why would I do this?
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        coarse_labels = np.copy(labels.numpy())
        coarse_labels = map_subclasses(coarse_labels)
        coarse_labels = torch.from_numpy(coarse_labels)
        images, labels, coarse_labels = Variable(images).cuda(), Variable(labels).cuda(), Variable(coarse_labels).cuda()
        #myim = np.transpose(images[12].cpu(),(1,2,0))
        #myim = myim/2 + 0.5
        #plt.imshow(myim)
        #plt.show()
        #sys.exit(1)
        coarse_out, fine_out = net(images)
        top_prob_c, top_label_c = torch.topk(coarse_out, 20)
        top_prob_f, top_label_f = torch.topk(fine_out, 20)
        #print(top_prob_c, top_label_c)
        _, predicted_coarse = torch.max(coarse_out.data, 1)
        _, predicted_fine = torch.max(fine_out.data, 1)
        total += labels.size(0)
        top_label_c1 = top_label_c[:,0]
        top_prob_c1 = top_prob_c[:,0]
        top_prob_c2 = top_prob_c[:,1]
        top_prob_c3 = top_prob_c[:,2]
        top_prob_f1 = top_prob_f[:,0]
        #print(top_prob_c1, top_prob_f1)
        #sys.exit(1)
        top_label_c2 = top_label_c[:,1]
        top_label_c3 = top_label_c[:,2]
        top_label_f1 = top_label_f[:,0]
        top_label_f2 = top_label_f[:,1]
        top_label_f3 = top_label_f[:,2]
            
        correct_coarse += (top_label_c1 == coarse_labels.data).sum()
        correct_coarse2 = correct_coarse2 + (top_label_c2 == coarse_labels.data).sum()
        correct_coarse3 = correct_coarse3 + (top_label_c3 == coarse_labels.data).sum()
        check1 = (top_label_c1 == coarse_labels.data)
        top_fine = top_prob_f[:,0]
        
        num_iter = list(top_fine.size())
        for i in range(0,num_iter[0]):
            for j in range(0,20):
                b = top_label_c[i,0].cpu().numpy()
                a = top_label_f[i,j].cpu().numpy()
                a = mmap[a]
         #       print(i, a, b)
                if(b == a):
                    break
            top_fine[i] = top_prob_f[i,j]
        '''
        for i in range(0,len(top_fine)):
            print(top_fine[i]
        '''
        #divider = (top_prob_f1 / top_prob_c1)
        divider = (top_fine / top_prob_c1)
        check2 = (divider < alpha) 
        check1 = check1.int()
        check2 = check2.int()
        check1[check1 == 0] = -1
        if(plotter == 1):
            for j in range(0,16):
                real_coarse = int(coarse_labels.data[j])
                real_coarse = coarse[real_coarse]
                cl_g1_idx = int(top_label_c1[j])
                cl_g1 = coarse[cl_g1_idx]
                cl_g1_p = float(top_prob_c1[j])

                cl_g2_idx = int(top_label_c2[j])
                cl_g2 = coarse[cl_g2_idx]
                cl_g2_p = float(top_prob_c2[j])


                cl_g3_idx = int(top_label_c3[j])
                cl_g3 = coarse[cl_g3_idx]
                cl_g3_p = float(top_prob_c3[j])


                f1_g1_idx = int(top_label_f1[j])
                f1_g1 = fine[f1_g1_idx]
                f1_g1_p = float(top_prob_f1[j])
                if(check1[j] == check2[j]):
                    f1_g1 = "unk"
                    f1_g1_p = 0.000000
                title = ('''actual coarse label: %s,
                coarse guess 1: %s -- percentage likelihood: %.5f
                coarse guess 2: %s -- percentage likelihood: %.5f
                coarse guess 3: %s  -- percentage likelihood: %.5f
                fine guess: %s  -- percentage likelihood: %.5f
                ''' % (real_coarse, cl_g1, cl_g1_p, cl_g2, cl_g2_p, cl_g3, cl_g3_p, f1_g1, f1_g1_p))
                
                
                myim = np.transpose(images[j].cpu(),(1,2,0))
                myim = myim/2 + 0.5
               # plt.figure(1, figsize = (6,6))
               # plt.title(title, fontsize = 10) 
               # plt.imshow(myim)
               # plt.tight_layout()
               # plt.show()


        unk_coarse += (check1 == check2).sum() 
        #unk_coarse += ((top_label_c1 == coarse_labels.data) and (top_prob_f1*2 < top_prob_c1)).sum() 
        #unk_coarse += (top_label_c1 == coarse_labels.data)and(top_prob_f1 < (top_prob_c1 - (alpha * top_prob_c1))).sum() 
        correct_fine += (top_label_f1 == labels.data).sum()
        correct_fine2 = correct_fine2 + (top_label_f2 == labels.data).sum()
        correct_fine3 = correct_fine3 + (top_label_f3 == labels.data).sum()
        fine_loss = criterion1(fine_out, labels)
        coarse_loss = criterion2(coarse_out, coarse_labels)
        total_loss_fine += fine_loss.item()
        total_loss_coarse += coarse_loss.item()
    net.train() # Why would I do this?
    correct_coarse2 += correct_coarse
    correct_coarse3 += correct_coarse2
    correct_fine2 += correct_fine
    correct_fine3 += correct_fine2
    return (correct_coarse.float() / total, correct_fine.float()/ total, unk_coarse.float()/total,
            correct_coarse2.float() / total, correct_fine2.float()/ total,
            correct_coarse3.float() / total, correct_fine3.float()/ total)

def split_train(fine, coarse, i):
    test = []
    train = []
    #for i in range(0,20):
    a = np.where(mmap == i)
    b = np.random.choice(a[0], 1, replace=False)
    a = a[0]
    b = int(b)
    a = a.tolist()
    a.remove(b)
    for k in a:
        train.append(k)
    test.append(b)

    return train, test
def split_train2(fine, coarse, i):
    test = []
    train = []
    #for i in range(0,20):
    a = np.where(mmap == i)
    b = np.random.choice(a[0], 1, replace=False)
    a = a[0]
    b = int(b)
    a = a.tolist()
    for k in a:
        train.append(k)
    test.append(b)

    return train, test
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 250 #maximum epoch to train

    classes = pickle.load(open('./data/cifar-100-python/meta', 'rb'))
    fine = classes['fine_label_names']
    coarse = classes['coarse_label_names'] 
    a = []
    for name in fine:
        i = 0
        for sublist in classes_c2f_map:
            if name in sublist:
                a.append(i)
            i = i + 1
    mmap = np.array(a)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32, scale=(0.75,1),ratio=(0.75,1.25)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    #print(trainset.data)

    #classes = ('plane', 'car', 'bird', 'cat',
    #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net_dict = net.state_dict()
    #optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.0007, momentum = 0.85)
    pretrained = torch.load('block5_branch.pth')
    pretrained = {k: v for k, v in pretrained['state_dict'].items() if k in net_dict}
    net_dict.update(pretrained)
    net.load_state_dict(net_dict)
    #optimizer.load_state_dict(pretrained['optimizer'])
    net.train() # Why would I do this?

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    alpha_arr = []
    for superclass in range(0,20):
        train_l, test_l = split_train(fine,coarse,superclass)
        testset1 = CIFAR100(root='./data', idcs_in = train_l, train=False,
                download=True, transform=transform)
        testloader1 = torch.utils.data.DataLoader(testset1, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=2)
        testset2 = CIFAR100(root='./data', idcs_in = test_l, train=True,
                download=True, transform=transform)
        testloader2 = torch.utils.data.DataLoader(testset2, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=2)

        net.eval() # evaluation mode
        delta = 0.5
        points = []
        alp = []
        alpha = 0.25
        prev_avg_false = 1
        while(True):
            #print('________________________________________________________________________________')
            train_acc1_c, train_acc1_f, unk, train_acc2_c, train_acc2_f, train_acc3_c, train_acc3_f = eval_net2(testloader1,0,alpha)
            #print('Test on known testing labels\nCOARSE             FINE\ntest_acc1c: %.5f test_acc1f: %.5f\ntest_acc2c: %.5f  test_acc2f: %.5f\ntest_acc3c: %.5f  test_acc3f: %.5f\n'%
            #      (train_acc1_c, train_acc1_f, train_acc2_c, train_acc2_f, train_acc3_c, train_acc3_f))
            TU = 1 - (float(unk) / (float(train_acc1_c.cpu().numpy())))
            #print("True knowns:", TU)
            TU_adj = (float(train_acc1_c.cpu().numpy()) * TU)
            train_acc1_c, train_acc1_f, unk, train_acc2_c, train_acc2_f, train_acc3_c, train_acc3_f = eval_net2(testloader2,0,alpha)
            #print('Test on unknown testing labels\nCOARSE             FINE\ntest_acc1c: %.5f test_acc1f: %.5f\ntest_acc2c: %.5f  test_acc2f: %.5f\ntest_acc3c: %.5f  test_acc3f: %.5f\n'%
            #      (train_acc1_c, train_acc1_f, train_acc2_c, train_acc2_f, train_acc3_c, train_acc3_f))
            TK = (float(unk) / float(train_acc1_c.cpu().numpy()))
            TK_adj = (float(train_acc1_c.cpu().numpy()) * TK)
            #print("True unknowns:", TK)
            #print(alpha, "avg false:", (TK+TU)/2)
            alp.append(alpha)
            points.append((TU+TK)/2)


            prev_avg_false = (TK+TU)/2
            alpha = alpha+0.01
            if(alpha > 0.99):
                break
        points = np.array(points)
        idx = np.argmax(points)
        print(superclass, alp[idx])
        alpha_arr.append(alp[idx])
    print(alpha_arr)
    alpha_arr = np.array(alpha_arr)
    np.savetxt("alpha1.csv", alpha_arr, delimiter=",")


