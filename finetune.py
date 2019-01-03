# fine tune the vgg model using datasets with small learning rate hence the filter doesnt get changed too much
import os
import torch
import torchvision
import copy
import torchvision.transforms as transforms
# Variable is a torch specific data structure
from torch.autograd import Variable
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as model
from util import plot_confusion_matrix
from sklearn import svm



# places class to get the train test loader
def places(seed):
	np.random.seed(seed)
	preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize,])
	# create imagefolder 
	places_train = torchvision.datasets.ImageFolder('./places/train', transform = preprocessor)
	places_train.train_labels=[]
	# extract all the labels for train instances
	for i in range(places_train.__len__()):
		places_train.train_labels.append(places_train.imgs[i][1])
	# create loader class to load formated data
	train_loader = torch.utils.data.DataLoader(places_train, batch_size =1, shuffle=True, num_workers = 1, pin_memory = True)
	
	# create test folder
	places_test = torchvision.datasets.ImageFolder('./places/test', transform = preprocessor)
	places_test.test_labels = []
	# extract all the labels for test instances
	for i in range(places_test.__len__()):
		places_test.test_labels.append(places_test.imgs[i][1])
	# create loader class to load test data 
	test_loader = torch.utils.data.DataLoader(places_test, batch_size =8, shuffle = True, num_workers = 1, pin_memory=True)
	
	# return train and test loader
	return train_loader, test_loader

# resize the image to desired dimensions
resize = transforms.Resize((224,224))
#resize = transforms.Resize((299,299))
# normalize the image to desired mean and std
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
								std = [0.229, 0.224, 0.225])
# define preprocessor
preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize,])
train_loader, test_loader = places(1)
# size is the first dimension of shape
dataset_size = np.shape(train_loader.dataset.train_labels)[0]
testset_size = np.shape(test_loader.dataset.test_labels)[0]

	
# now define the function to train model
def train_model(model, criterion, optimizer, train_loader, scheduler, num_epochs):
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        training_loss = 0.0
        training_correct = 0.0
		# iterate over the data, mimic the example codes:
        for i, data in enumerate(train_loader, 0):
            print(i)
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            print("inputs")
            #print(input)
            print("labels")
            print(labels)
			# zero the parameter gradients first
            optimizer.zero_grad()
			# get forward pass result
            outputs = model.forward(inputs)
            print("outputs")
            #print(outputs)
            _, preds = torch.max(outputs.data,1)
            print("prediction")
            print(preds)
			# obtain loss given labels, outputs and loss rule
            loss = criterion(outputs, labels)
			# now do back propagation(calculate gradients)
            loss.backward()
			# update parameter based on optimizing rule
            optimizer.step()
            print("loss.data")
            print(loss.data)
			# get numbers
            training_loss += loss.item()
            training_correct += (preds == labels.data).sum()
            print("training loss")
            #print(training_loss)
            print("training correct")
            #print(training_correct)

		# now obtain epoch loss and accuracy
        epoch_loss = training_loss*1.0/(i+1)
        epoch_accuracy = training_correct*100.0/(i+1)
        
        print("Loss is: {:.4f} Accuracy is: {:.4f}".format(epoch_loss, epoch_accuracy))
		# copy the model
        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            best_model = copy.deepcopy(model.state_dict())
        time_elapsed = time.time()-start
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed*1.0/60, time_elapsed%60))
        print("best accuracy: {:4f}".format(best_acc))
        
	# now load the best model weights
    model.load_state_dict(best_model)
    return model
    
		
# function to test the model
def test_model(model, test_loader):
    correct_num = 0.0
    y_predict = []
    y_actual =[]
    for iter, data in enumerate(test_loader, 0):
        # get the inputs
        inputs, labels = data
		# make them variables
        
        # if use GPU
        inputs = Variable(inputs)
        labels = Variable(labels)
        
		# forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data,1)
        correct_num += (preds ==labels.data).sum()
        y_predict += list(preds)
        y_actual += list(labels.data)

    test_accuracy = correct_num*100.0/testset_size
    print("test accuracy: {:.4f}".format(test_accuracy))
    plot_confusion_matrix(y_predict, y_actual, "VGG16_fintuned")
    return y_predict


# main functinon for VGG16
def main_vgg():
	#transfer learning, load the pretrained vgg model
    pretrained_model = model.vgg16(pretrained =True)
	# now modify the last fully connected layer to match the dimension of places
    print("loaded pretrained model")
	#extract the list of the model and modify
    pretrained_model_list = list(pretrained_model.classifier.children())
	# pop the last layer
    pretrained_model_list.pop()
    # add the final layer
    pretrained_model_list.append(nn.Linear(4096, 9))
	# now put the new classifier into the model
    new_classifier = nn.Sequential(*pretrained_model_list)
    pretrained_model.classifier = new_classifier
    print("modified network")
    # start training model
    pretrained_model.train()
	# use cross entropy as loss
    criterion = nn.CrossEntropyLoss()
	# setup optimization
    optimizer_vgg16 = optim.SGD(pretrained_model.parameters(), lr = 0.001, momentum =0.9)
	# use an exponentially decaying learning rate
    scheduler = lr_scheduler.StepLR(optimizer_vgg16, step_size =6, gamma =0.01)
    best_model = train_model(pretrained_model, criterion, optimizer_vgg16, train_loader, scheduler, 6)
    y_predict = test_model(best_model, test_loader)
    plt.show()

# define function to train inception net
def main_inception():
    # load the inception nn model 
    pretrained_model = model.resnet18(pretrained=True)
    # inception doesnt have classifier attribute hence directly modify
    pretrained_model.fc = nn.Linear(512, 9)
    pretrained_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_inception = optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_inception, step_size =7, gamma=0.1)
    best_model = train_model(pretrained_model, criterion, optimizer_inception, train_loader, scheduler, 6)
    y_pred = test_model(best_model, test_loader)
    plt.show()


#main_vgg()
main_inception()

