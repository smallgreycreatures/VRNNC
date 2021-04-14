# Adapted from VRNN implemented by p0werHu: https://github.com/p0werHu/VRNN

import  torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from utils import Progbar
from model import VRNN
from loss_function import loss as Loss
from mnist_config import Config

def load_dataset(batch_size):
    """
    train_dataset = datasets.MNIST(root='mnist_data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='mnist_data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    """
    dataset = datasets.MNIST(root='mnist_data', train=True, download=True,
        transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=4, shuffle=True, pin_memory=True,
    )

    dataset2 = datasets.MNIST('mnist_data', train=False,
                       transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)
    return train_loader, test_loader

def train(conf):

    train_loader, test_loader = load_dataset(512)
    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Start training with device: ",device)
    torch.cuda.manual_seed_all(112858)
    model.to(device)
    model = torch.nn.DataParallel(model)
    if conf.restore == True:
        model.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
        print('Restore model from ' + conf.checkpoint_path)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_loss = 0
    for epoch in range(1, conf.train_epoch + 1):
        training_loss = 0
        recon_loss = 0
        kld_loss = 0
        classification_loss = 0
        
        for i, (data, target) in enumerate(train_loader):
            correct = 0
            data = data.squeeze(1)
            #data = (data / 255).to(device)
            data = data.to(device)
            target = target.to(device)
            #print(data.shape)
            #data = data.to(device)
            package = model(data)
            #prior_means, prior_var, decoder_means, decoder_var, x_decoded = package
            #print(len(prior_means))
            loss,rec_loss,kl_loss,class_loss = Loss(package, data, target)

            for pred_label in package[-1]:
 
                correct += pred_label.eq(target.data.view_as(pred_label)).sum()/28

            model.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            training_loss += loss.item()
            recon_loss +=rec_loss.item()
            kld_loss += kl_loss.item()
            classification_loss += class_loss.item()/512
        print('Train Epoch: {} \t Loss: {:.6f}, NLL: {:.6f}, KLD: {:.6f}, Classfier: {:.6f}, Correct: {:.6f}'.format(
                epoch,
                training_loss,
                recon_loss,
                kld_loss,
                classification_loss,
                correct))
        with torch.no_grad():
            x_decoded = model.module.sampling(conf.x_dim, device)
            x_decoded = x_decoded.cpu().numpy()
            digit = x_decoded.reshape(conf.x_dim, conf.x_dim)

            plt.imshow(digit, cmap='Greys_r')
            plt.pause(1e-6)

        if epoch % conf.save_every == 0:
            torch.save(model.state_dict(), 'Epoch_' + str(epoch + 1) + '.pth')


def generating(conf):

    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
    print('Restore model from ' + conf.checkpoint_path)

    with torch.no_grad():
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        for i in range(n):
            for j in range(n):
                x_decoded = model.module.sampling(digit_size, device)
                x_decoded = x_decoded.cpu().numpy()
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()



if __name__ == '__main__':

    conf = Config()
    train(conf)
    generating(conf)





