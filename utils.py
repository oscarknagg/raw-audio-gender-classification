import torch


def whiten(batch,rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    m = batch.mean(dim=1)
    transform = batch-m.repeat([batch.shape[1], 1]).transpose(dim0=1,dim1=0)
    transform = transform*rms/torch.abs(transform).mean(dim=1).repeat([batch.shape[1], 1]).transpose(dim0=1, dim1=0)
    return transform


def evaluate(model, dataloader, preprocessor):
    """
    This function evaluates the performance of a model on a dataset. I will use this to determine when the model reaches
    peak generalisation performance and save the weights.
    :param model: Model to evaluate
    :param dataloader: An instance of a pytorch DataLoader class
    :param preprocessor: Function that takes a batch and performs any required preprocessing
    :return: Accuracy of the model on the data supplied by dataloader
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            batch, labels = data

            batch = preprocessor(batch)

            predicted = model(batch)
            total += labels.size(0)
            correct += ((predicted > 0.5)[:, 0] == labels.cuda().byte()).cpu().sum().numpy()

    return correct * 1.0 / total
