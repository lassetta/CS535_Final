from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm import tqdm


classes_c2f_map = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                   ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                   ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                   ['bottle', 'bowl', 'can', 'cup', 'plate'],
                   ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                   ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                   ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                   ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                   ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                   ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                   ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                   ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                   ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                   ['crab', 'lobster', 'snail', 'spider', 'worm'],
                   ['baby', 'boy', 'girl', 'man', 'woman'],
                   ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                   ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                   ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                   ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                   ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
mmap = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # auxiliary
        self.dropout_conv = nn.Dropout(p=0.15)
        self.dropout_fc = nn.Dropout(p=0.25)
        # block 1
        self.conv_b1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_b1_bn1 = nn.BatchNorm2d(32)
        self.conv_b1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_b1_bn2 = nn.BatchNorm2d(32)
        self.conv_b1_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_b1_bn3 = nn.BatchNorm2d(32)
        self.pool_b1 = nn.MaxPool2d(2, 2)
        # block 2
        self.conv_b2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_b2_bn1 = nn.BatchNorm2d(64)
        self.conv_b2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_b2_bn2 = nn.BatchNorm2d(64)
        self.conv_b2_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_b2_bn3 = nn.BatchNorm2d(64)
        self.pool_b2 = nn.MaxPool2d(2, 2)
        # block 3
        self.conv_b3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_b3_bn1 = nn.BatchNorm2d(128)
        self.conv_b3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_b3_bn2 = nn.BatchNorm2d(128)
        self.conv_b3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_b3_bn3 = nn.BatchNorm2d(128)
        self.pool_b3 = nn.MaxPool2d(2, 2)
        # block 4
        self.conv_b4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv_b4_bn1 = nn.BatchNorm2d(256)
        self.conv_b4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_b4_bn2 = nn.BatchNorm2d(256)
        self.conv_b4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_b4_bn3 = nn.BatchNorm2d(256)
        self.pool_b4 = nn.MaxPool2d(2, 2)
        # block 5
        self.conv_b5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_b5_bn1 = nn.BatchNorm2d(512)
        self.conv_b5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_b5_bn2 = nn.BatchNorm2d(512)
        self.conv_b5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_b5_bn3 = nn.BatchNorm2d(512)
        self.pool_b5 = nn.MaxPool2d(2, 2)

        # coarse branch 1 -> generalized classes
        self.br1_fc1 = nn.Linear(512, 4096)
        self.br1_bn1 = nn.BatchNorm1d(4096)
        self.br1_fc2 = nn.Linear(4096, 4096)
        self.br1_bn2 = nn.BatchNorm1d(4096)
        self.br1_fc3 = nn.Linear(4096, 20)
        self.br1_sm = nn.Softmax(dim=1)
        # branch 2 -> Fine layers
        self.br2_fc1 = nn.Linear(512, 4096)
        self.br2_bn1 = nn.BatchNorm1d(4096)
        self.br2_fc2 = nn.Linear(4096, 4096)
        self.br2_bn2 = nn.BatchNorm1d(4096)
        self.br2_fc3 = nn.Linear(4096, 100)
        self.br2_sm = nn.Softmax(dim=1)

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
        br1_x = x.view(-1, self.num_flat_features(x))
        br1_x = self.br1_fc1(br1_x)
        br1_x = self.br1_bn1(br1_x)
        br1_x = F.leaky_relu(br1_x)
        br1_x = self.dropout_fc(br1_x)
        br1_x = self.br1_fc2(br1_x)
        br1_x = self.br1_bn2(br1_x)
        br1_x = F.leaky_relu(br1_x)
        br1_x = self.dropout_fc(br1_x)
        br1_x = self.br1_fc3(br1_x)
        br1_out = self.br1_sm(br1_x)
        # offshoot branch 2
        br2_x = x.view(-1, self.num_flat_features(x))
        br2_x = self.br2_fc1(br2_x)
        br2_x = self.br2_bn1(br2_x)
        br2_x = F.leaky_relu(br2_x)
        br2_x = self.dropout_fc(br2_x)
        br2_x = self.br2_fc2(br2_x)
        br2_x = self.br2_bn2(br2_x)
        br2_x = F.leaky_relu(br2_x)
        br2_x = self.dropout_fc(br2_x)
        br2_x = self.br2_fc3(br2_x)
        br2_out = self.br2_sm(br2_x)
        return(br1_out, br2_out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def map_subclasses(array):
    for i in range(0, len(array)):
        array[i] = mmap[array[i]]
    return(array)


def eval_net(dataloader):
    correct_coarse = 0
    correct_fine = 0
    total = 0
    total_loss_coarse = 0
    total_loss_fine = 0
    net.eval()
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        coarse_labels = np.copy(labels.numpy())
        coarse_labels = map_subclasses(coarse_labels)
        coarse_labels = torch.from_numpy(coarse_labels)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        coarse_labels = Variable(coarse_labels).cuda()
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
    net.train()
    return (total_loss_coarse / total, correct_coarse.float() / total,
            total_loss_fine / total, correct_fine.float() / total)


def objective(model, input, coarse_label, fine_labels):
    coarse_out, fine_out = model(input)
    # return torch.max(fine_out[0, fine_labels])
    return 2 * torch.max(fine_out[0, fine_labels]) - coarse_out[0, coarse_label]


def objective_gradient(objective, model, input, coarse_label, fine_labels):
    x = input.detach().clone()
    x.requires_grad = True
    output = objective(model, x, coarse_label, fine_labels)
    output.backward()
    grad = x.grad.clone()
    return grad


def projection(input):
    input = input
    input[input > 1] = 1
    input[input < -1] = -1
    return input


def attack(model, input, coarse_label, fine_labels, epsilon):
    curr = input.clone()
    curr.requires_grad = True

    for iter in range(30):

        grad = objective_gradient(objective, model, curr,
                                  coarse_label, fine_labels)
        grad = epsilon * grad / torch.norm(grad, p="fro")
        next = curr - grad

        curr.data = next.clone()

    return projection(curr.detach())


def job(i, model, images, labels):
    image = images[i, :, :, :].unsqueeze(0)
    label = labels[i]
    coarse_label = map_subclasses(np.array(label, ndmin=2))
    fine_labels = np.flatnonzero(mmap == coarse_label)
    coarse_label = torch.from_numpy(coarse_label)
    fine_labels = torch.from_numpy(fine_labels)

    output_image = attack(net, image, coarse_label, fine_labels, 1)

    return (output_image, coarse_label)


if __name__ == "__main__":
    BATCH_SIZE = 32  # mini_batch size

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

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
    net = Net().cpu()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.85)
    pretrained = torch.load('block5_branch.pth', map_location=torch.device('cpu'))
    net.load_state_dict(pretrained['state_dict'])
    net.eval()

    output_labels = np.empty((0, 1))
    output_images = np.empty((0, 3, 32, 32))

    # set up multiprocessing
    num_cores = mp.cpu_count()

    ctr = 0
    for data in trainloader:
        images, labels = data
        n_items = len(labels)
        idxs = tqdm(range(n_items))
        results = (Parallel(n_jobs=num_cores)(delayed(job)(i, net, images, labels) for i in idxs))

        for i in range(n_items):
            output_images = np.append(output_images, results[i][0].numpy(), axis=0)
            output_labels = np.append(output_labels, results[i][1].numpy(), axis=0)

        np.save("attacked_images.npy", output_images)
        np.save("attacked_image_labels.npy", output_labels)
