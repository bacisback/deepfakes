
import torch
import numpy as np

from torch.autograd import Variable

def train_epoch(model, train_loader, criterion, optimizer, device):

    # put model in train mode
    model.train()

    # keep track of the training losses during the epoch
    train_losses = []

    for batch in train_loader:

        # Move the training data to the GPU
        inputs = Variable(batch['X'])
        labels = Variable(batch['Y'])

        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)

        inputs, labels = inputs.to(device), labels.to(device)

        # clear previous gradient computation
        optimizer.zero_grad()

        # forward propagation
        _, predictions = model(inputs)

        # calculate the lossfor the batch
        loss = criterion(predictions, labels)
        # backpropagate to compute gradients
        loss.backward()
        # update model weights
        optimizer.step()

        # update the array of batch losses
        train_losses.append(loss.item())

    # calculate average training loss of the epch and return
    return sum(train_losses) / len(train_losses)



def validate(model, valid_loader, criterion, device):

    # put model in evaluation mode
    model.eval()

    # keep track of losses and predictions
    valid_losses = []
    y_pred = []
    y_labl = []

    # We don't need gradients for validation, so wrap in
    # no_grad to save memory
    with torch.no_grad():

        for batch in valid_loader:

            # Move the validation batch to the GPU
            inputs = Variable(batch['X'])
            labels = Variable(batch['Y'])

            labels = labels.unsqueeze(1)
            labels = labels.to(torch.float32)

            inputs, labels = inputs.to(device), labels.to(device)

            # forward propagation
            # predictions, interm_feats = ???
            _, predictions = model(inputs)

            # calculate the loss
            loss = criterion(predictions, labels)

            # update running loss value
            valid_losses.append(loss.item())

            #print(predictions)
            #print(predictions.argmax(dim=1))
            #print(np.round(predictions, 0))
            #print('----------------------')
            #print(labels)

            # save predictions
            y_pred.extend(np.round(predictions, 0).flatten())
            y_labl.extend(labels.cpu().numpy().flatten())

    # compute the average validation loss
    valid_loss = sum(valid_losses) / len(valid_losses)

    # Collect predictions into y_pred and ground truth into y_true
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_labl, dtype=np.float32)

    # Calculate accuracy as the average number of times y_true == y_pred
    accuracy = (y_pred == y_true).sum().item() / len(y_true)

    return valid_loss, accuracy



def train(model, train_loader, valid_loader, criterion, optimizer, epochs, first_epoch=0, validation=True, device='cuda'):
    '''Train the model for a specific number of epochs.
    '''
    train_losses, valid_losses = [],  []

    for epoch in range(first_epoch, first_epoch + epochs):

        # training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if validation:
            # validation phase
            valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

            print(f'[{epoch:03d}] train loss: {train_loss:04f}  '
                  f'val loss: {valid_loss:04f}  '
                  f'val acc: {valid_acc*100:.4f}%')

            valid_losses.append(valid_loss)

        else:

            print(f'[{epoch:03d}] train loss: {train_loss:04f}  ')


        train_losses.append(train_loss)

        # Save a checkpoint
        checkpoint_filename = f'checkpoints/mnist-{epoch:03d}.pkl'
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)

    return train_losses, valid_losses


def save_checkpoint(optimizer, model, epoch, filename):

    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }

    torch.save(model, filename)




if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()

    # create an SGD optimizer with learning rate 0.01, momentum 0.9, and nesterov momentum turned on
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
