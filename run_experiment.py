from scipy.signal import resample
import time

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from config import PATH, LIBRISPEECH_SAMPLING_RATE
from data import LibriSpeechDataset
from models import *
from utils import whiten, evaluate


print('Training {} GPU support'.format('with' if torch.cuda.is_available() else 'without'))


##############
# Parameters #
##############
n_seconds = 3
downsampling = 1
batchsize = 8
model_type = 'max_pooling'
model_n_layers = 7
model_n_filters = 64
model_dilation_depth = 7  # Only relevant for model_type == 'dilated'
model_dilation_stacks = 1  # Only relevant for model_type == 'dilated'
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
learning_rate = 0.005
momentum = 0.9
n_epochs = 10
reduce_lr_patience = 32
evaluate_every_n_batches = 500


# Generate model name. The name should contain enough information to determine the shape of the weights matrices and
# the shape of the input
if model_type == 'max_pooling':
    model_name = 'max_pooling__n_layers={}__n_filters={}__downsampling={}__n_seconds={}.torch'.format(
        model_n_layers, model_n_filters, downsampling, n_seconds
    )
elif model_type == 'dilated':
    model_name = 'dilated__n_depth={}__n_stacks={}__n_filters={}__downsampling={}__n_seconds={}.torch'.format(
        model_dilation_depth, model_dilation_stacks, model_n_filters, downsampling, n_seconds
    )
else:
    raise(ValueError, 'Model type not recognised.')


def preprocessor(batch):
    batch = whiten(batch)
    batch = torch.from_numpy(
        resample(batch, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
    ).reshape((batchsize, 1, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
    return batch


###################
# Create datasets #
###################
trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)
trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size=batchsize, num_workers=4, drop_last=True)


################
# Define model #
################
if model_type == 'max_pooling':
    model = ConvNet(model_n_filters, model_n_layers)
elif model_type == 'dilated':
    model = DilatedNet(model_n_filters, model_dilation_depth, model_dilation_stacks)
else:
    raise(ValueError, 'Model type not recognised.')
model.double()
model.cuda()


#############################
# Define loss and optimiser #
#############################
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


#################
# Training loop #
#################
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=reduce_lr_patience)

best_accuracy = 0
val_acc_values = []
acc_values = []

t0 = time.time()

print('\n[Epoch, Batches, Seconds]')
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_correct_samples = 0
    for i, data in enumerate(trainloader, 0):
        # Get batch
        inputs, labels = data

        # Normalise the volume to a fixed root mean square value as some speakers are much quieter than others
        inputs = whiten(inputs)

        # Resample audio
        inputs = torch.from_numpy(
            resample(inputs, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
        ).reshape((batchsize, 1, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.reshape((batchsize, 1)).cuda().double())
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        running_correct_samples += torch.eq((outputs[:, 0] > 0.5).cpu(), labels.byte()).numpy().sum()
        if i % evaluate_every_n_batches == evaluate_every_n_batches - 1:  # print every 'print_every' mini-batches
            val_acc = evaluate(model, testloader, preprocessor)

            # return model to training mode
            model.train()

            print('[%d, %5d, %.1f] loss: %.3f acc: %.3f val_acc: %.3f' %
                  (epoch + 1, i + 1, time.time() - t0,
                   running_loss / evaluate_every_n_batches,
                   running_correct_samples * 1. / (evaluate_every_n_batches * batchsize),
                   val_acc))
            running_loss = 0.0
            running_correct_samples = 0

            val_acc_values.append(val_acc)
            acc_values.append((running_correct_samples * 1. / (evaluate_every_n_batches * batchsize)))

            # Save new model if its the best
            if val_acc > best_accuracy:
                print('Saving new best model.')
                torch.save(model.state_dict(), PATH + '/models/' + model_name)
                best_accuracy = val_acc

            # Check for plateau
            scheduler.step(val_acc)

print('\nFinished Training')
print('Best validation accuracy was {:.3f}'.format(best_accuracy))
print('Best model weights saved to: {}'.format(PATH + '/models/' + model_name))
