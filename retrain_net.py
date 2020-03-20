from __future__ import print_function
from __future__ import division
import numpy as np
import pickle
import torch
import os
from torch.autograd import Variable
from cifar10 import CIFAR100
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys
#from tensorboardX import SummaryWriter  # for pytorch below 1.14

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
        self.bran1_fc1 = nn.Linear(512, 4096)
        self.bran1_bn1 = nn.BatchNorm1d(4096)
        self.bran1_fc2 = nn.Linear(4096,4096)
        self.bran1_bn2 = nn.BatchNorm1d(4096)
        self.bran1_fc3 = nn.Linear(4096,20)
        self.bran1_sm = nn.Softmax(dim=1)
        # branch 2 -> Fine layers
        self.bran2_fc1 = nn.Linear(512, 4096)
        self.bran2_bn1 = nn.BatchNorm1d(4096)
        self.bran2_fc2 = nn.Linear(4096,4096)
        self.bran2_bn2 = nn.BatchNorm1d(4096)
        self.bran2_fc3 = nn.Linear(4096,100)
        self.bran2_sm = nn.Softmax(dim=1)


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
        bran1_x = x.view(-1, self.num_flat_features(x))
        bran1_x = self.bran1_fc1(bran1_x)
        bran1_x = self.bran1_bn1(bran1_x)
        bran1_x = F.leaky_relu(bran1_x)
        bran1_x = self.dropout_fc(bran1_x) 
        bran1_x = self.bran1_fc2(bran1_x)
        bran1_x = self.bran1_bn2(bran1_x)
        bran1_x = F.leaky_relu(bran1_x)
        bran1_x = self.dropout_fc(bran1_x) 
        bran1_x = self.bran1_fc3(bran1_x)
        # offshoot branch 2
        bran2_x = x.view(-1, self.num_flat_features(x))
        bran2_x = self.bran2_fc1(bran2_x)
        bran2_x = self.bran2_bn1(bran2_x)
        bran2_x = F.leaky_relu(bran2_x)
        bran2_x = self.dropout_fc(bran2_x) 
        bran2_x = self.bran2_fc2(bran2_x)
        bran2_x = self.bran2_bn2(bran2_x)
        bran2_x = F.leaky_relu(bran2_x)
        bran2_x = self.dropout_fc(bran2_x) 
        bran2_x = self.bran2_fc3(bran2_x)
        return(bran1_x,bran2_x)


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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    BATCH_SIZE = 32 #mini_batch size

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
        for k in a:
            train.append(k)
        test.append(b)
    return train, test

def translate_to_unknown(lab, unknown_set):
    print(unknown_set)
    return lab

if __name__ == "__main__":
    classes = pickle.load(open('./data/cifar-100-python/meta', 'rb'))
    fine = classes['fine_label_names']
    coarse = classes['coarse_label_names'] 

    torch.manual_seed(0)
    np.random.seed(0)
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 250 #maximum epoch to train
    a = []
    for name in fine:
        i = 0
        for sublist in classes_c2f_map:
            if name in sublist:
                a.append(i)
            i = i + 1
    mmap = np.array(a)


    train_l, test_l = split_train(fine,coarse)

    print(test_l)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32, scale=(0.9,1.1),ratio=(0.9,1.1)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    attacked_im = np.load("attacked_images.npy", allow_pickle = True)
    attacked_im1 = np.load("attacked_images1.npy", allow_pickle = True)
    attacked_im2 = np.load("attacked_images2.npy", allow_pickle = True)
    attacked_lab = np.load("attacked_image_labels.npy", allow_pickle = True)
    attacked_lab1 = np.load("attacked_image_labels1.npy", allow_pickle = True)
    attacked_lab2 = np.load("attacked_image_labels2.npy", allow_pickle = True)


    attacked_im_cat = np.concatenate((attacked_im,attacked_im1,attacked_im2), axis = 0)
    attacked_lab_cat = np.concatenate((attacked_lab,attacked_lab1,attacked_lab2), axis = 0)
    attecked_im = None
    attecked_im1 = None
    attecked_im2= None
    attecked_lab = None
    attecked_lab1 = None
    attecked_lab2 = None
    a,_ = attacked_lab_cat.shape
    attacked_lab_cat = np.reshape(attacked_lab_cat,(a))
    attacked_lab_cat = attacked_lab_cat.astype(int)
    iterator = 0
    for fine_idx in test_l:
        if iterator == 0:
            remove_idcs = np.argwhere(attacked_lab_cat == fine_idx)
            iterator = 1
        else:
            idcs_remove_single = np.argwhere(attacked_lab_cat == fine_idx)
            remove_idcs = np.concatenate((remove_idcs,idcs_remove_single),axis=0)
    remove_idcs = np.reshape(remove_idcs, (-1)) 
    
    print(attacked_im_cat.shape)
    print(attacked_lab_cat.shape)
    attacked_lab_cat = np.delete(attacked_lab_cat, remove_idcs, axis=0)
    attacked_im_cat = np.delete(attacked_im_cat, remove_idcs, axis=0)
    print("fine")
    print(attacked_im_cat.shape)
    print(attacked_lab_cat.shape)
    print(np.bincount(attacked_lab_cat))
    print("to coarse")
    attacked_lab_cat = map_subclasses(attacked_lab_cat) 
    print(attacked_lab_cat.shape)
    print(np.bincount(attacked_lab_cat))
    print(np.unique(attacked_lab_cat))
    print("to unknown")
    attacked_lab_cat = np.array([test_l[i] for i in attacked_lab_cat])
    print(attacked_lab_cat.shape)
    print(np.bincount(attacked_lab_cat))
    print(np.unique(attacked_lab_cat))

    print("Loading data")
    atk_lab_ten = torch.Tensor(attacked_lab_cat)
    print(atk_lab_ten)
    sys.exit(1)
    attacked_lab_cat = None
    atk_im_ten = torch.Tensor(attacked_im_cat)
    attacked_im_cat = None
    atk_dset = torch.utils.data.TensorDataset(atk_im_ten, atk_lab_ten)
    atk_dload = torch.utils.data.DataLoader(atk_dset, batch_size = BATCH_SIZE, shuffle = True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    testset1 = CIFAR100(root='./data', idcs_in = train_l, train=True,
                download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(testset1, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=2)
    print("Loading model")
    net = Net().cuda()
    net_dict = net.state_dict()
    pretrained = torch.load('extra.pth')
    pretrained = {k: v for k, v in pretrained['state_dict'].items() if k in net_dict}
    net_dict.update(pretrained)
    net.load_state_dict(net_dict)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    print("beginning the training")
    
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        net.train() # training mode
        running_loss = 0.0
        if epoch < 50:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.85)
        elif epoch < 100:
            optimizer = optim.SGD(net.parameters(), lr=0.005, momentum = 0.85)
        elif epoch < 150:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.85)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum = 0.85)
            
        if epoch < 30:
            alpha = 0.58
            beta = 0.42
        else:
            alpha = 0.9
            beta = 0.1
        for i, data in enumerate(atk_dload, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.long()
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
        print('    Finish training this EPOCH, start evaluating...')
        net.eval() # evaluation mode
        train_loss_c, train_acc_c, train_loss_f, train_acc_f = eval_net(trainloader)
        test_loss_c, test_acc_c, test_loss_f, test_acc_f = eval_net(testloader)


        print('EPOCH: COARSE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_c, train_acc_c, test_loss_c, test_acc_c))

        print('EPOCH: FINE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_f, train_acc_f, test_loss_f, test_acc_f))

        #writer.add_scalar('train_loss', train_loss)
        #writer.add_scalar('test_loss', test_loss)
        if os.path.exists("block5_branch.pth"):
            os.remove("block5_branch.pth")
        state = {
            'state_dict': net.state_dict(),
            }
        torch.save(state, 'block5_branch.pth')
        f = open("phaseRT_results.csv", 'a')
        f.write("%d, %.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n"%
              (epoch+1, train_loss_c, train_acc_c, test_loss_c, test_acc_c,
              train_loss_f, train_acc_f, test_loss_f, test_acc_f))
        f.close()

    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'block5_retrained.pth')

