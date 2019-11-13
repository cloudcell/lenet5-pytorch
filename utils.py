import torch


def model_checker(model, dataset, device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device).long()
    # We have to turn batchnorm off as it mixes the samples in the batch.
    use_bn = model.use_bn
    model.use_bn = False
    chart_dependencies(model, data)
    model.use_bn = use_bn


def chart_dependencies(model, data):
    """Use backprop to chart dependencies in the model. This function checks
    That the model does not mix the input samples.
    """
    N = data.shape[0]

    data.requires_grad = True
    model.train()
    model.zero_grad()
    output = model(data)

    for i in range(N):
        loss = output[i, ::].sum()
        loss.backward(retain_graph=True)
        for j in range(N):
            if j != i:
                assert data.grad[j, ::].max() == 0.
        data.grad.zero_()
