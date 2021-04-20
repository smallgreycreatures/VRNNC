# Adapted from VRNN implemented by p0werHu: https://github.com/p0werHu/VRNN

import  torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from utils import Progbar
from VRNN_model import VRNN
from VRNN_loss_function import loss as Loss
from eeg_config import Config
from process_data_rnn_ssm import min_max_scaler,sliding_window




def pre_process_data(all_data,test_data,labels,test_labels,mat,nasty_num=1,sub=1,w=60):
    trials = mat['data_eeg'][0][0][5]
    indicies = np.where(trials == nasty_num)
    trial = indicies[0][0]
    #print(indicies[0].shape)
    for i,trial in enumerate(indicies[0]):
        data = mat['data_eeg'][0][0][1][0][trial]

        sub_sampled = data.T[1::sub]
        #print(sub_sampled.shape)


            
        scaler = preprocessing.StandardScaler().fit(sub_sampled)
        scaled = scaler.transform(sub_sampled)
        scaled = min_max_scaler(scaled)
        #plt.plot(scaled[:,0:3])
        sw_data = sliding_window(scaled,w=w)
        #print(sw_data.shape)
        #sw_data2 = sw_data.reshape((72,30,355))
        if len(indicies[0])>10 and i >10 and i <16 and len(test_data):
            test_data = np.append(test_data,sw_data,axis=0)
            
            if nasty_num == 1:
                test_labels = np.append(test_labels, sw_data.shape[0])
            else:
                test_labels = np.append(test_labels,np.ones(sw_data.shape[0]))
        else:
            all_data = np.append(all_data,sw_data, axis=0)
            if nasty_num == 1:
                labels = np.append(labels, sw_data.shape[0])
            else:
                labels = np.append(labels,np.ones(sw_data.shape[0]))
    return all_data,test_data, labels,test_labels

def load_data(ground_path,smell,w=380,sub_sample=4,cols=72,train_test_ratio=10):
    dir_names = [name for name in os.listdir(ground_path) if os.path.isdir(name) and name.startswith('SL') ]
    dir_names.sort()
    ind = 0
    all_data = np.zeros((1,w,cols))

    hold_out_person_data = np.zeros((1,w,cols))
    for i, dir_name in enumerate(dir_names):
        path = ground_path+dir_name
        exp_names = os.listdir(path)
        exp_names.sort()
        #print(exp_names)
        
            
        
        for exp_name in exp_names:
            full_path= path+"/"+exp_name
            #print(full_path)
            mat = scipy.io.loadmat(full_path)
            all_data = pre_process_data(all_data,mat,smell,sub_sample,w)
        if i > 15:
            for exp_name in exp_names:
                full_path= path+"/"+exp_name
                #print(full_path)
                mat = scipy.io.loadmat(full_path)
                hold_out_person_data = pre_process_data(hold_out_person_data,mat,smell,sub_sample,w)

    train_data = all_data[1:]

    hold_out_person_data = hold_out_person_data[1:]
    if smell == 1:
        train_labels = np.zeros(train_data.shape[0])
        hold_out_person_labels = np.zeros(hold_out_person_data.shape[0])
    else:
        test_labels = np.ones(train_data.shape[0])
        hold_out_person_labels = np.ones(hold_out_person_data.shape[0])


    n = int(disgust_data.shape[0]/train_test_ratio)
    test_data = np.zeros((n,w,cols))
    test_labels = np.zeros(n)
    for i in range(0,n):
        index = np.random.randint(0,disgust_data.shape[0])
        e = train_data[index]
        l = train_labels[index]
        train_data = np.delete(train_data,index,axis=0)
        train_labels = np.delete(train_labels,index)
        test_data[i] = e
        test_labels[i] = l

    return train_data,train_labels,test_data,test_labels,hold_out_person_data,hold_out_person_labels


def load_dataset(batch_size,ground_path,new_data,w=380,sub_sample=4,cols=72,train_test_ratio=10):
    if new_data:
        print("Loading in new data...")
        pleasant_train,pleasant_labels,pleasant_test,pleasant_labels,pleasant_hold_out,pleasant_hold_out_labels = load_data(ground_path,1)
        disgust_train,disgust_labels,disgust_test,disgust_labels,disgust_hold_out,disgust_hold_out_labels = load_data(ground_path,32)

        train_data = np.append(pleasant_train,disgust_train,axis=0)
        train_labels = np.append(pleasant_labels,disgust_labels)
        test_data = np.append(pleasant_test,disgust_test,axis=0)
        test_labels = np.append(pleasant_labels,disgust_labels)
        hold_out_person_data = np.append(pleasant_hold_out,disgust_hold_out,axis=0)
        hold_out_person_labels = np.append(pleasant_hold_out_labels,disgust_hold_out_labels)

        with open('train_data.npy', 'wb') as f:
            np.save(f,train_data)
        with open('train_labels.npy', 'wb') as f:
            np.save(f,train_labels)
        with open('test_data.npy', 'wb') as f:
            np.save(f,test_data)
        with open('test_labels.npy', 'wb') as f:
            np.save(f,test_labels)
        with open('hold_out_person_data.npy', 'wb') as f:
            np.save(f,hold_out_person_data)
        with open('hold_out_person_labels.npy', 'wb') as f:
            np.save(f,hold_out_person_labels)
    else:
        print("Loading stored data...")
        train_data = np.load(ground_path+"train_data.npy")
        train_labels = np.load(ground_path+"train_labels.npy")
        test_data = np.load(ground_path+"test_data.npy")
        test_labels = np.load(ground_path+"test_labels.npy")
        hold_out_person_data = np.load(ground_path+"hold_out_person_data.npy")
        hold_out_person_labels = np.load(ground_path+"hold_out_person_labels.npy")
 
    train_dataset = TensorDataset(torch.Tensor(train_data),torch.Tensor(train_labels))
    train_dataloader = DataLoader(train_dataset,batch_size=512,shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(test_data),torch.Tensor(test_labels))
    test_dataloader = DataLoader(test_dataset,batch_size=512,shuffle=True)

    ho_dataset = TensorDataset(torch.Tensor(hold_out_person_data),torch.Tensor(hold_out_person_labels))
    ho_dataloader = DataLoader(train_dataset,batch_size=512,shuffle=True)

    return train_dataloader, test_dataloader, ho_dataloader

def train(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cpu'):
        ground_path=conf.ground_path_cpu
    else:
        ground_path=conf.ground_path_gpu


    train_loader, test_loader,ho_loader = load_dataset(512,ground_path,False)
    print("Data locked and loaded!")
    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    
    print("Start training with device: ",device)
    torch.cuda.manual_seed_all(112858)
    model.to(device)
    model = torch.nn.DataParallel(model)
    if conf.restore == True:
        model.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
        print('Restore model from ' + conf.checkpoint_path)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_loss = 0
    losses = np.zeros((conf.train_epoch,3))
    for epoch in range(1, conf.train_epoch + 1):
        training_loss = 0
        recon_loss = 0
        kld_loss = 0

        for i, (data, target) in enumerate(train_loader):
            data = data.squeeze(1)
            #data = (data / 255).to(device)
            data = data.to(device)
            target = target.to(device)
            #print(data.shape)
            #data = data.to(device)
            package = model(data)
            #prior_means, prior_var, decoder_means, decoder_var, x_decoded = package
            #print(len(prior_means))
            loss,rec_loss,kl_loss = Loss(package, data)
            model.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            training_loss += loss.item()
            recon_loss +=rec_loss.item()
            kld_loss += kl_loss.item()
        print('Train Epoch: {} \t Loss: {:.6f}, NLL: {:.6f}, KLD: {:.6f}'.format(
                epoch,
                training_loss,
                recon_loss,
                kld_loss))
        losses[epoch-1] = np.array([training_loss,recon_loss,kld_loss])
        if epoch % conf.save_every == 0:
            torch.save(model.state_dict(), 'Epoch_' + str(epoch + 1) + '.pth')
    with open('losses.npy', 'wb') as f:
        np.save(f,losses)
def generating(conf):

    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
    print('Restore model from ' + conf.checkpoint_path)

    with torch.no_grad():
        sequence_length = 380
        x_decoded = model.module.sampling(sequence_length, device)
        x_decoded = x_decoded.cpu().numpy()

def classify(conf):
    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.checkpoint_path, map_location='cuda:0'))
    print('Restore model from ' + conf.checkpoint_path)

def generate_sequences():
    ground_path='/Users/corytrevor/Documents/Skola/KTH/EE/Master/exjobb/Code/VRNNC/data/numpy_neuro_data/'

    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoints/2021-04-14-VRNN_EEG_Epoch_301.pth",map_location=torch.device('cpu')))
    train_data = np.load(ground_path+"train_data.npy")
    train_labels = np.load(ground_path+"train_labels.npy")

    disgust_data = torch.from_numpy(train_data).float()
    decoded_data = np.zeros((disgust_data.shape))
    with torch.no_grad():
        package = model.forward(disgust_data)
        for i, time_period in enumerate(package[-1]):
            decoded_data[:,i,:] = time_period
        with open('sequence.npy', 'wb') as f:
            np.save(f,decoded_data)

def get_dec_means():
    ground_path='/Users/corytrevor/Documents/Skola/KTH/EE/Master/exjobb/Code/VRNNC/data/numpy_neuro_data/'

    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoints/2021-04-14-VRNN_EEG_Epoch_301.pth",map_location=torch.device('cpu')))
    train_data = np.load(ground_path+"train_data.npy")
    train_labels = np.load(ground_path+"train_labels.npy")
    disgust_data = torch.from_numpy(train_data).float()
    decoded_data = np.zeros((disgust_data.shape[0],380,16))
    with torch.no_grad():
        package = model.forward(disgust_data)
        for i, time_period in enumerate(package[2]):
            decoded_data[:,i,:] = time_period
        with open('dec_means_train_data.npy', 'wb') as f:
            np.save(f,decoded_data)
    
def get_z():
    ground_path='/Users/corytrevor/Documents/Skola/KTH/EE/Master/exjobb/Code/VRNNC/data/numpy_neuro_data/'

    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoints/2021-04-14-VRNN_EEG_Epoch_301.pth",map_location=torch.device('cpu')))
    train_data = np.load(ground_path+"train_data.npy")
    train_labels = np.load(ground_path+"train_labels.npy")

    disgust_data = torch.from_numpy(train_data).float()
    decoded_data = np.zeros((disgust_data.shape[0],disgust_data.shape[1],16))
    with torch.no_grad():
        package = model.forward(disgust_data)
        for i, time_period in enumerate(package[-1]):
            #print(decoded_data.shape)
            #print(time_period.shape)
            decoded_data[:,i,:] = time_period
        with open('z.npy', 'wb') as f:
            np.save(f,decoded_data)
if __name__ == '__main__':

    conf = Config()
    train(conf)
    #generate_sequences()
    get_z()
    #get_dec_means()





