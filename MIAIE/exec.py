import random
import torch.nn as nn
import torch.optim as optim
import util.mol_conv as mc
import util.trainer as tr
from torch_geometric.data import DataLoader
from model.GAT import GAT


max_epochs = 500


dataset = mc.read_dataset('data/example.csv')
num_train_mols = int(len(dataset) * 0.9)
random.shuffle(dataset)
train_dataset = dataset[:num_train_mols]
test_dataset = dataset[num_train_mols:]


train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32)


model = GAT(mc.num_atom_feats, mc.num_mol_feats, 1).cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)


for epoch in range(0, max_epochs):
    train_loss = tr.train(model, train_data_loader, criterion, optimizer)
    test_loss = tr.test(model, test_data_loader, criterion)
    print('Epoch {}\ttrain loss {:.4f}\ttest loss {:.4f}'.format(epoch + 1, train_loss, test_loss))

tr.estimate_ai(model, test_dataset)
