from __future__ import print_function
from __future__ import division
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter  # for pytorch below 1.14
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # auxiliary
        self.dropout_conv = nn.Dropout(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.8)
        # block 1
        self.conv_b1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_b1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool_b1 = nn.MaxPool2d(2,2)
        # block 2
        self.conv_b2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_b2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool_b2 = nn.MaxPool2d(2,2)
        # coarse branch 1 -> generalized classes
        self.br1_fc1 = nn.Linear(64 * 8 * 8, 4096)
        self.br1_bn1 = nn.BatchNorm1d(4096)
        self.br1_fc2 = nn.Linear(4096,20)
        # block 3
        self.conv_b3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_b3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool_b3 = nn.MaxPool2d(2,2)
        # branch 2 -> Fine layers
        self.br2_fc1 = nn.Linear(128 * 4 * 4, 4096)
        self.br2_bn1 = nn.BatchNorm1d(4096)
        self.br2_fc2 = nn.Linear(4096,100)


    def forward(self, x):
        # block 1 forward prop
        x = F.leaky_relu(self.conv_b1_1(x))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b1_2(x))
        x = self.dropout_conv(x)
        x = self.pool_b1(x)
        # block 2 forward prop
        x = F.leaky_relu(self.conv_b2_1(x))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b2_2(x))
        x = self.dropout_conv(x)
        x = self.pool_b2(x)
        # offshoot branch 1
        br1_x = x.view(-1, self.num_flat_features(x))
        br1_x = self.br1_fc1(br1_x)
        br1_x = self.br1_bn1(br1_x)
        br1_x = F.leaky_relu(br1_x)
        br1_x = self.dropout_fc(br1_x) 
        br1_x = self.br1_fc2(br1_x)
        # block 3 forward prop
        x = F.leaky_relu(self.conv_b3_1(x))
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.conv_b3_2(x))
        x = self.dropout_conv(x)
        x = self.pool_b3(x)
        # offshoot branch 2
        br2_x = x.view(-1, self.num_flat_features(x))
        br2_x = self.br2_fc1(br2_x)
        br2_x = self.br2_bn1(br2_x)
        br2_x = F.leaky_relu(br2_x)
        br2_x = self.dropout_fc(br2_x) 
        br2_x = self.br2_fc2(br2_x)
        return(br1_x,br2_x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def map_subclasses(array):
    f = lambda t: int(t/5)
    ret_arr = np.vectorize(f)
    array = ret_arr(array)
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
        coarse_labels = labels.numpy()
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
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 150 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat',
    #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    coarse_dict = [int(i/5) for i in range(0,100)]
    print('Building model...')
    net = Net().cuda()
    net.train() # Why would I do this?

    writer = SummaryWriter(log_dir='./log')
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay = 5e-4)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        net.train() # training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            coarse_labels = labels.numpy()
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
            loss = (0.5)*loss1 + (0.5)*loss2
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
        test_loss_c, test_acc_c, test_loss_f, test_acc_f = eval_net(testloader)

        print('EPOCH: COARSE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_c, train_acc_c, test_loss_c, test_acc_c))

        print('EPOCH: FINE -> %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss_f, train_acc_f, test_loss_f, test_acc_f))
        #writer.add_scalar('train_loss', train_loss)
        #writer.add_scalar('test_loss', test_loss)
        writer.add_scalars('loss', {'test': test_loss_c, 'train': train_loss_c},epoch)
        writer.add_scalars('accuracy', {'test': test_acc_c, 'train': train_acc_c},epoch)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'strict_test.pth')
