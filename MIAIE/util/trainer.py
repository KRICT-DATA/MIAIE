import torch


def train(model, data_loader, criterion, optimizer):
    model.train()
    train_loss = 0

    for i, (bg) in enumerate(data_loader):
        bg.batch = bg.batch.cuda()
        preds = model(bg)
        loss = criterion(bg.y, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()

    with torch.no_grad():
        test_loss = 0

        for i, (bg) in enumerate(data_loader):
            bg.batch = bg.batch.cuda()
            preds = model(bg)
            loss = criterion(bg.y, preds)
            test_loss += loss.detach().item()

        return test_loss / len(data_loader)


def estimate_ai(model, dataset):
    model.eval()
    model.est(True)

    with torch.no_grad():
        for mol in dataset:
            preds, rgsa1, rgsa2, rgsa3 = model.compute_rgsa(mol)

            print('-----------------')
            print('Prediction error: ' + str(torch.abs(preds - mol.y)))
            print('Mol Id: ' + str(mol.id))
            print(mol.atom_nums)
            print(mol.edge_index)
            print(rgsa1)
            print(rgsa2)
