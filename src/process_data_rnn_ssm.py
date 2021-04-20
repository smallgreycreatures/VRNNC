import pickle
import numpy as np
import os, os.path

#from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data_help(prev_stop,stop,path,final_data,c=3,sub_sample=1,w=60):
    all_data = np.zeros((stop,84*3))
    for i in range(0,stop):
        if i < 10 and stop >= 1000:
            filepath = path+'000' +str(i)
        elif i < 10:
            filepath = path+'00' +str(i)
        elif i >= 10 and i <100 and stop >= 1000:
            filepath = path+'00' +str(i)
        elif i >= 10 and i <100:
            filepath = path+'0' +str(i)
        elif i >= 100 and i <1000 and stop >= 1000:
            filepath = path+'0' +str(i)
        elif i >= 100 and i <1000:
            filepath = path+str(i)
        elif i>=1000:
            filepath = path+str(i)
        #print(filepath)
        with open(filepath+'.tbnd') as f:
            data = [[float(num) for num in line.split(' ')] for line in f]
        data = np.array(data).flatten()
        all_data[i]=data
    if sub_sample:
        all_data = all_data[1::4]

    scaled_data = min_max_scaler(all_data)
    Y = np.zeros((scaled_data.shape[0],2))
    if c == 0:
        for i in range(Y.shape[0]):
            Y[i,0] = 1
        scaled_class_data = np.append(scaled_data,Y,axis=1)
    elif c == 1:
        for i in range(Y.shape[0]):
            Y[i,1] = 1
        scaled_class_data = np.append(scaled_data,Y,axis=1)
    else:
        scaled_class_data = scaled_data
    sw_data = sliding_window(scaled_class_data)
    if sub_sample:
        final_data[prev_stop:prev_stop+sw_data.shape[0]] = sw_data
    else:
        final_data[prev_stop:prev_stop+stop-w] = sw_data
    return final_data,sw_data.shape[0]

def min_max_scaler(all_data):
    scaler = MinMaxScaler()
    scaler.fit(all_data)
    scaled_data = scaler.transform(all_data)
    return scaled_data

def sliding_window(scaled_data,w=30,s=1,t=0):
    n_column = scaled_data.shape[1]
    limit_r = int((scaled_data.shape[0]-t-w+1)/s + 1)
    period = []
    slide_window_data = np.zeros((limit_r-1,w,n_column))
    i = 0
    for r in range(1, limit_r):
        period.append(f'{t+(r-1)*s}~{t+(r-1)*s+w-1}')
        #print(SlideWindowDF)

        slide_window_data[i,:,:] = scaled_data[t+(r-1)*s: t+(r-1)*s+w]
        i+=1

    return slide_window_data

def one_hot(Y,num):
    for i in range(Y.shape[0]):
        Y[i,num] = 1
    return Y
def gen_data_help(path,w):

    DIR1 = path +"T1/"
    DIR2 = path +"T2/"
    stop = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])-1
    stop8 = len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))])-1
    final_data = np.zeros((int((stop+stop8)/4-w*2+1),w+1,84*3))
    final_data,prev_stop = load_data_help(0,stop,DIR1,final_data,3,w=w)
    Y_1 = np.zeros((int(np.ceil(stop/4)),2))
    Y_1 = one_hot(Y_1,0)
    Y_1_sw = sliding_window(Y_1)

    final_data,prev_stop = load_data_help(prev_stop,stop8,DIR2,final_data,3,w=w)
    Y_2 = np.zeros((int(np.ceil(stop8/4)),2))
    Y_2 = one_hot(Y_2,1)
    Y_2_sw = sliding_window(Y_2)
    final_data_xy = np.zeros((int((stop+stop8)/4-w*2+1),w+1,84*3+2))
    final_data_xy,prev_stop = load_data_help(0,stop,DIR1,final_data_xy,0,w=w)
    final_data_xy,prev_stop = load_data_help(prev_stop,stop8,DIR2,final_data_xy,1,w=w)
    Y_train = np.vstack((Y_1[0:-w],Y_2[0:-w]))
    Y_train_sw = np.vstack((Y_1_sw,Y_2_sw))
    return  final_data, final_data_xy,Y_train,Y_train_sw

def generate_data(num,w):
    ground_path='/Users/corytrevor/Documents/Skola/KTH/EE/Master/exjobb/Code/binghamton3D/new3DFeatures'

    dir_names = os.listdir(ground_path)
    dir_names.sort()
    w = w-1
    for i,n in enumerate(dir_names[1:num]):
        path = ground_path+"/"+n+"/"
        final_data, final_data_xy,Y_train1,Y_train_sw1 = gen_data_help(path,w)

        if i == 0:
            sw_data = final_data[:,0,:]
            sw_data_x = final_data
            sw_data_xy = final_data_xy
            Y_train = Y_train1
            Y_train_sw = Y_train_sw1
        else:
            sw_data = np.concatenate((sw_data,final_data[:,0,:]))
            sw_data_x = np.concatenate((sw_data_x,final_data))
            sw_data_xy = np.concatenate((sw_data_xy,final_data_xy))
            Y_train = np.concatenate((Y_train,Y_train1))
            Y_train_sw = np.vstack((Y_train_sw,Y_train_sw1))
    """
    w = 59
    path = '/F001/'
    final_data1, final_data_xy1,Y_train1,Y_train_sw1 = gen_data_help(path,w)

    path = '/F002/'
    final_data2, final_data_xy2,Y_train2,Y_train_sw2 = gen_data_help(path,w)
    path = '/F003/'
    final_data3, final_data_xy3,Y_train3,Y_train_sw3 = gen_data_help(path,w)
    path = '/F004/'
    final_data4, final_data_xy4,Y_train4,Y_train_sw4 = gen_data_help(path,w)
    path = '/F005/'
    final_data5, final_data_xy5,Y_train5,Y_train_sw5 = gen_data_help(path,w)

    print(Y_train5.shape)
    sw_data = np.concatenate((final_data1[:,0,:],final_data2[:,0,:],final_data4[:,0,:],final_data5[:,0,:]))
    sw_data_x = np.concatenate((final_data1[:,:,:],final_data2[:,:,:],final_data4[:,:,:],final_data5[:,:,:]))
    sw_data_xy = np.concatenate((final_data_xy1[:,:,:],final_data_xy2[:,:,:],final_data_xy4[:,:,:],final_data_xy5[:,:,:]))
    print(sw_data.shape)
    Y_train = np.concatenate((Y_train1,Y_train2,Y_train4,Y_train5))
    Y_train_sw = np.vstack((Y_train_sw1,Y_train_sw2,Y_train_sw4,Y_train_sw5))
    #Y_train = Y_train[1::4]
    print(Y_train.shape)
    """
    return sw_data, sw_data_x,sw_data_xy,Y_train,Y_train_sw