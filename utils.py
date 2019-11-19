import os
import torch

from visualizer import PredVisualizer


def model_checker(model, dataset, device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device).long()
    chart_dependencies(model, data)


def chart_dependencies(model, data):
    """Use backprop to chart dependencies in the model. This function checks
    that the model does not mix the samples in an input batch.
    """
    N = data.shape[0]

    data.requires_grad = True
    model.eval()
    model.zero_grad()
    output = model(data)

    for i in range(N):
        loss = output[i, ::].sum()
        loss.backward(retain_graph=True)
        for j in range(N):
            if j != i:
                assert data.grad[j, ::].max() == 0.
        data.grad.zero_()


def save_checkpoint(model, optimizer, epoch, save_dir):
    checkpoint_path = os.path.join(save_dir, 'checkpoint_' + str(epoch) + '.pt')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])


def eval(model, batch, device):
    """Run inference on a batch during training."""
    model.eval()
    X, y = batch
    X, y = X.to(device), y.to(device).long()
    preds = model(X)
    model.train()
    return preds


def init_vis(dataset ,path, batch_size=16):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    vis = PredVisualizer(batch, path)
    return vis
