# Adapted from VRNN implemented by p0werHu: https://github.com/p0werHu/VRNN


import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
import torch


def loss(package, x):

    prior_means, prior_var, decoder_means, decoder_var, x_decoded,classification_loss = package
    loss = 0.
    for i in range(x.shape[1]):
        # Kld loss
        norm_dis1 = Norm.Normal(prior_means[i], prior_var[i])
        norm_dis2 = Norm.Normal(decoder_means[i], decoder_var[i])
        kld_loss = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))

        # reconstruction loss
        xent_loss = torch.mean(F.binary_cross_entropy(x_decoded[i], x[:, i, :], reduction='none'))
        loss += xent_loss + kld_loss

    return loss+classification_loss,xent_loss,kld_loss,classification_loss