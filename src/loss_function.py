# Adapted from VRNN implemented by p0werHu: https://github.com/p0werHu/VRNN


import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
import torch


def loss(package, x,labels):

    prior_means, prior_var, decoder_means, decoder_var, x_decoded,all_classified,_ = package
    loss = 0.
    classification_loss = 0
    kld_loss_total = 0
    nll_loss_total = 0
    for i in range(x.shape[1]):
        # Kld loss
        norm_dis1 = Norm.Normal(prior_means[i], prior_var[i])
        norm_dis2 = Norm.Normal(decoder_means[i], decoder_var[i])
        kld_loss = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))
        #print(kld_loss)
        # reconstruction loss
        nll_loss = torch.mean(F.binary_cross_entropy(x_decoded[i], x[:, i, :], reduction='none'))
        #print(xent_loss)
        loss += nll_loss + kld_loss
        #print(all_classified[i].shape,labels.shape)
        kld_loss_total +=kld_loss
        nll_loss_total += nll_loss
        classification_loss+= F.nll_loss(all_classified[i],labels)
    return loss+classification_loss/512,nll_loss_total,kld_loss_total,classification_loss/512