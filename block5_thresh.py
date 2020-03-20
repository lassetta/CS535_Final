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

def split_train(fine, coarse):
    test = []
    train = []
    for i in range(0,20):
        a = np.where(mmap == i)
        b = np.random.choice(a[0], 1, replace=False)
        a = a[0]
        b = int(b)
        a = a.tolist()
        a.remove(b)
        print(a,b)
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
    train_l, test_l = split_train(fine,coarse)
    print(train_l, test_l)
    print(len(train_l), len(test_l))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32, scale=(0.75,1),ratio=(0.75,1.25)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = CIFAR100(root='./data', idcs_in = train_l, train=True,
            download=True, transform=train_transform)
    print(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = CIFAR100(root='./data', idcs_in = test_l, train=True,
            download=True, transform=transform)
    print(testset)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    #print(trainset.data)

    #classes = ('plane', 'car', 'bird', 'cat',
    #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net_dict = net.state_dict()
    #optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.0007, momentum = 0.85)
    pretrained = torch.load('extra.pth')
    pretrained = {k: v for k, v in pretrained['state_dict'].items() if k in net_dict}
    net_dict.update(pretrained)
    print(pretrained)
    net.load_state_dict(net_dict)
    #optimizer.load_state_dict(pretrained['optimizer'])
    net.train() # Why would I do this?

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        net.train() # training mode
        running_loss = 0.0
        if(epoch < 50):
            alpha = 0.6
            beta = 0.4
        if(epoch < 25):
            alpha = 0.1
            beta = 0.9
        else:
            alpha = 0.9
            beta = 0.1
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            coarse_labels = np.copy(labels.numpy())
            coarse_labels = map_subclasses(coarse_labels)
            coarse_labels = torch.from_numpy(coarse_labels)

            # wrap them in Variable
            inputs, labels, coarse_labels = Variable(inputs).cuda(), Variable(labels).cuda(), Variable(coarse_labels).cuda()
            coarse_out, fine_out = net(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss1 = criterion1(fine_out, labels)
            loss2 = criterion2(coarse_out, coarse_labels)
            loss = alpha*loss1 + beta*loss2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            '''
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
            '''
        print('    Finish training this EPOCH, start evaluating...')
        net.eval() # evaluation mode
        train_loss_c, train_acc_c, train_loss_f, train_acc_f = eval_net(trainloader)
        print('EPOCH: COARSE -> %d train_loss: %.5f train_acc: %.5f' %
              (epoch+1, train_loss_c, train_acc_c))

        print('EPOCH: FINE -> %d train_loss: %.5f train_acc: %.5f' %
              (epoch+1, train_loss_f, train_acc_f))
        '''
        test_loss_c, test_acc_c, test_loss_f, test_acc_f = eval_net(testloader)
        print('EPOCH: COARSE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_c, train_acc_c, test_loss_c, test_acc_c))

        print('EPOCH: FINE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_f, train_acc_f, test_loss_f, test_acc_f))
        '''
        #writer.add_scalar('train_loss', train_loss)
        #writer.add_scalar('test_loss', test_loss)
        if os.path.exists("extra.pth"):
            os.remove("extra.pth")
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(state, 'extra.pth')

    print('Finished Training')
    print('Saving model...')
