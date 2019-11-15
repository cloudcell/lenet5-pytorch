import torch


def model_checker(model, dataset, device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device).long()
    chart_dependencies(model, data)


def chart_dependencies(model, data):
    """Use backprop to chart dependencies in the model. This function checks
    That the model does not mix the samples in an input batch.
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
