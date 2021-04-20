# implement by p0werHu
# time 11/18/2019


class Config(object):

    def __init__(self):
        # define some configures here
        self.x_dim = 72
        self.h_dim = 100
        self.z_dim = 8
        self.train_epoch = 3
        self.save_every = 10
        self.batch_size = 512
        self.device_ids = [0, 1]
        self.checkpoint_path = '../checkpoint/Epoch_61.pth'
        self.restore = False
        self.ground_path_cpu = '/Users/corytrevor/Documents/Skola/KTH/EE/Master/exjobb/Code/VRNNC/data/numpy_neuro_data/'
        self.ground_path_gpu = '/local_storage/datasets/fnorden/numpy_neuro_data/'
