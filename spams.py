from dataset import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
torch.manual_seed(3)

batch_size =120
hidden_dim = 10
embedding_dim = 8
vocab_size = 28049
target_size = 2
epochs = 1

# function to embed the sentences
def extraction(seed):
	np.random.seed(seed)
	embedding = nn.Embedding(vocab_size, embedding_dim)
	# embed the training samples
	train_iterator, test_iterator = load_dataset("spam", vocab_size)
	for batch in train_iterator:
		sentence = batch.sentence
		label_train = batch.label
		embedded_train = embedding(sentence)
	# embed the test samples
	for batch in test_iterator:
		sentence = batch.sentence
		label_test = batch.label
		embedded_test = embedding(sentence)
	print(embedded_train.size())
	print(label_train.size())
	print(embedded_test.size())
	print(label_test.size())
	return embedded_train, label_train, embedded_test, label_test
	
# define GRU class

class GRU(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
		super(GRU, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim)
		
		# last linear layer to map the output of GRU to classes
		self.linear = nn.Linear(hidden_dim, target_size)
		self.hidden = self.hidden_init()
		
	# define a function to initialize a hidden state 
	def hidden_init(self):
	# initialize the hidden state
		#return (Variable(torch.zeros(1, batch_size, self.hidden_dim)), Variable(torch.zeros(1, batch_size, self.hidden_dim)))
		return Variable(torch.zeros(1, batch_size, hidden_dim))
	# define forward function
	def forward(self, sentence, labels):
		embeds = self.word_embeddings(sentence)
		#print("embeds")
		#print(embeds)
		#print(embeds.size())
		#print("expected input dimension")
		#print(embeds.view(embeds.size()[0], batch_size, embedding_dim).size())
		# obtain the output and hidden state
		gru_out, self.hidden = self.gru(embeds, self.hidden)
		#print("gru_out")
		#print(gru_out)
		#print("gruout size")
		#print(gru_out.size()) 
		#linear_dim = gru_out.size(0)*gru_out.size(2)
		#print("linear_dim")
		#print(linear_dim)
		
		
		#linear = torch.nn.Linear(gru_out.size(), 2)
		last_output = gru_out[-1,:,:]
		#last_output = gru_out
		#print("last_output")
		#print(last_output)
		
		#linear_output = linear(gru_out)
		#print("linear_output")
		#print(linear_output.size())
		spam_space = self.linear(last_output)
		#print("output size")
		#print(spam_space)
		#print("labels")
		#print(labels)
		loss = nn.CrossEntropyLoss(reduce = False)
		pred_label = loss(spam_space, labels)
		return spam_space, pred_label
		
# define main function
def main():
	
	# define GRU model
	model = GRU(embedding_dim, hidden_dim, vocab_size, target_size)
	# define optimizer
	optimizer = optim.Adam(model.parameters(), lr =0.005)
	# load train and test iterators
	train_iterator, test_iterator = load_dataset("spam", batch_size)

	# start training:
	model.train()
	# iterate through epochs
	for i in range(epochs):
		running_loss = 0
		j = 0
		for batch in train_iterator:
			#print(j)
			sentence = batch.sentence
			label = batch.label
			batch_updated = sentence.size()[1]
			label_updated = label.size()[0]
			if batch_updated != batch_size or label_updated != batch_size:
				continue
			# zero the gradient
			optimizer.zero_grad()
			#print("setence size")
			#print(sentence.size())
			#print("label size")
			#print(label.size())
			# compute the forward pass
			spam_space, y_pred = model.forward(sentence,label)
			#print("loss")
			#print(y_pred)
			# do back prop
			y_pred.sum().backward(retain_graph = True)
			# optimize
			optimizer.step()
			print("minibatch loss")
			print(y_pred.data[0])
			running_loss += y_pred.data[0]
		print("[Epoch: %d] loss: %.3f" % (i+1, running_loss))
		running_loss = 0.0
			
	with open("trained GRU for spam","wb") as fid:
		pickle.dump(model,fid)
	
	with open("trained GRU for spam","rb") as fid:
			model = pickle.load(fid)

	# now start testing
	correct = 0
	counter = 0
	model.eval()
	for batch in test_iterator:
		sentence = batch.sentence
		label = batch.label
		batch_updated = sentence.size()[1]
		label_updated = label.size()[0]
		if batch_updated != batch_size or label_updated!=batch_size:
			continue
		spam_space, y_prediction = model.forward(sentence, label)
		
		for i in range(batch_size):
			if spam_space.data[i][0]>spam_space.data[i][1]:
				prediction = 0
				counter+=1
			else:
				prediction = 1
			if prediction == label.data[0]:
				correct+=1
				counter+=1
	print("accuracy is:")
	print(correct, counter)
	print(correct*1.0/counter)


main()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
