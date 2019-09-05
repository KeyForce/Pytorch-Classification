import argparse
import os
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from torchvision.models.vgg import *
from models.vgg_test import *
from models import *
from utils import progress_bar
from visualzation.confusion_matrix import plot_confusion_matrix
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data.dataset import ConcatDataset

# 参数解析
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', default=True)
args = parser.parse_args()

# 设置GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 参数
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
classes = ('0', '1')  # 必须是数值型
classes_str = ['benign', 'suspicious']

# 定义自己的数据集 需要设计自己的解析代码
class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append(("/".join([words[0], words[1]]), words[2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(root + fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if label == 'benign':
            label = 0
        else:
            label = 1
        return img, label

    def __len__(self):
        return len(self.imgs)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_benign = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
])

transform_benign_two = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
])

transform_benign_three = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

root = '/home/hanwei-1/data/usg/ROI'

# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data = MyDataset(root, '/train.txt', transform=transform_train)
test_data = MyDataset(root, '/test.txt', transform=transform_test)
aug_benign = torchvision.datasets.ImageFolder(root='/home/hanwei-1/data/usg/ROI/Aug/',
                                              transform=transform_benign
                                              )
aug_benign_two = torchvision.datasets.ImageFolder(root='/home/hanwei-1/data/usg/ROI/Aug/',
                                              transform=transform_benign_two
                                              )
aug_benign_three = torchvision.datasets.ImageFolder(root='/home/hanwei-1/data/usg/ROI/Aug/',
                                              transform=transform_benign_three
                                              )

train_data = ConcatDataset([train_data, aug_benign, aug_benign_two, aug_benign_three])
train_test = ConcatDataset([train_data, test_data])

# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，loader的长度是有多少个batch，所以和batch_size有关
trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=64)
train_test_loader = DataLoader(dataset=train_test, batch_size=64, shuffle=True)
image = iter(aug_benign)

image, labels = next(image)
img = torchvision.utils.make_grid(image)
plt.imshow(img.numpy().transpose(1, 2, 0))
plt.show()


# 网络
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = vgg19_bn(num_classes=2)
# net = VGG('VGG19', 2)
net = vgg19_bn(num_classes=2)
# net = vgg19_bn(num_classes=2, init_weights=True, pretrained=False)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    cudnn.benchmark = True

if args.resume:
    # 加载 checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/vggckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def Test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    cm_targets = []
    cm_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            cm_targets.extend(targets.cpu().numpy())
            cm_predicted.extend(predicted.cpu().numpy())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    matrix = confusion_matrix(cm_targets, cm_predicted)
    plot_confusion_matrix(matrix, classes_str)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vggckpt.pth')
        best_acc = acc




for epoch in range(start_epoch, start_epoch + 10):
    # print(epoch)
    train(epoch)
    Test(epoch)
