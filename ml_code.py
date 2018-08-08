import numpy as np
import torch
from torchvision import transforms, datasets
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 64
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
data = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
num_train = len(data)
indices = list(range(num_train))
split = int(0.2 * num_train)
validation_idx = np.random.choice(indices, size=split, replace=False)

train_idx = list(set(indices) - set(validation_idx))
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, sampler=validation_sampler)

test_data = datasets.FashionMNIST('./data', train=False, transform=transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

class NeuralNet(nn.Module):
	def __init__(self,image_size):
		super(NeuralNet, self).__init__()
		self.image_size = image_size
		self.fc0 = nn.Linear(image_size, 100)
		self.fc1 = nn.Linear(100, 50)
		self.fc2 = nn.Linear(50, 10)
		self.name = "Regular"
	def forward(self, x):
		x = x.view(-1, self.image_size)
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return F.log_softmax(x, dim=1)

class NetWithDropout(nn.Module):
	def __init__(self,image_size):
		super(NetWithDropout, self).__init__()
		self.image_size = image_size
		self.fc0 = nn.Linear(image_size, 100)
		self.fc1 = nn.Linear(100, 50)
		self.fc2 = nn.Linear(50, 10)
		self.dropout = nn.Dropout(0.3)
		self.name = "Dropout"
	def forward(self, x):
		x = x.view(-1, self.image_size)
		x = F.relu(self.fc0(x))
		x = self.dropout(x)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		return F.log_softmax(x, dim=1)

class NetWithBatchNorm(nn.Module):
	def __init__(self,image_size):
		super(NetWithBatchNorm, self).__init__()
		self.image_size = image_size
		self.fc0 = nn.Linear(image_size, 100)
		self.fc0_bn = nn.BatchNorm1d(100)
		self.fc1 = nn.Linear(100, 50)
		self.fc1_bn = nn.BatchNorm1d(50)
		self.fc2 = nn.Linear(50, 10)
		self.name = "Batch Norm"
	def forward(self, x):
		x = x.view(-1, self.image_size)
		x = F.relu(self.fc0_bn(self.fc0(x)))
		x = F.relu(self.fc1_bn(self.fc1(x)))
		x = F.relu(self.fc2(x))
		return F.log_softmax(x, dim=1)


model_reg = NeuralNet(image_size=28*28)
optimizer_reg = optim.SGD(model_reg.parameters(), lr=0.01)
model_dr = NetWithDropout(image_size=28*28)
optimizer_dr = optim.SGD(model_dr.parameters(), lr=0.01)
model_bn = NetWithBatchNorm(image_size=28*28)
optimizer_bn = optim.SGD(model_bn.parameters(), lr=0.01)

models = [(model_reg, optimizer_reg), (model_dr, optimizer_dr), (model_bn, optimizer_bn)]



def train(model, opt):
	model.train()
	for batch_idx, (data, labels) in enumerate(train_loader):
		opt.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, labels)
		loss.backward()
		opt.step()

def validate(model):
	model.eval()
	total_loss = 0
	correct = 0

	# on validation
	for batch_idx, (data, target) in enumerate(validation_loader):
		output = model(data)
		total_loss += F.nll_loss(output, target, size_average=False).data.item()
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).double().cpu().sum()
	avg_loss = total_loss / len(validation_sampler)
	print('\nModel ' + model.name + ' Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_loss, correct, len(validation_sampler), 
																			100. * correct / len(validation_sampler)))
	#on training
	total_loss = 0
	correct = 0
	for batch_idx, (data, labels) in enumerate(train_loader):
		output = model(data)
		total_loss = F.nll_loss(output, labels, size_average=False).data.item()
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(labels.data.view_as(pred)).double().cpu().sum()
	avg_loss = total_loss / len(train_sampler)
	print('\nModel ' + model.name + ' Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_loss, correct, len(train_sampler), 
																			100. * correct / len(train_sampler)))
																			
def test(model):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\nModel ' + model.name + ' Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 
																			100. * correct / len(test_loader.dataset)))
def train_and_validate(model, opter):
	for epoch in range(1, 11):
		train(model, opter)
		validate(model)

for model, opt in models:
	train_and_validate(model, opt)

for model, opt in models:
	test(model)

