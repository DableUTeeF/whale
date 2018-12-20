from PIL import Image
import numpy as np
import os
from keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, csv,
                 rootpath,
                 input_size=None,
                 batch_size=None,
                 normalize=None,
                 logit=0,
                 dicider=True,
                 new_whale=True,
                 ):
        self.new_whale = new_whale
        if self.new_whale:
            self.csv = csv
        else:
            self.csv = []
            for i in csv:
                if i[1][0] != 'new_whale':
                    self.csv.append(i)
        self.rootpath = rootpath
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.batch_size = batch_size
        self.normalize = normalize if normalize is not None else lambda x: x
        self.logit = logit
        self.dicider = False if not new_whale else dicider
        self.getcls()

    def getcls(self):
        self.target = []
        for element in self.csv:
            self.target.append(element[1][0])
        self.target = sorted(list(set(self.target)))
        if self.dicider:
            self.target = self.target[1:]
        self.target_len = len(self.target)

    def get_single_image(self, idx):

        x = np.zeros((self.input_size[1], self.input_size[0], 3), dtype='float32')
        img = Image.open(os.path.join(self.rootpath, self.csv[idx][0])).resize(self.input_size).convert('RGB')
        x[:, :, :] = np.array(img, dtype='float32')
        if not self.batch_size:
            x = np.rollaxis(x, 2)
        x = self.normalize(x)
        z = np.ones(1, dtype='uint8')
        y = np.zeros(self.target_len, dtype='uint8')
        if self.csv[idx][1][0] == 'new_whale' and self.dicider:
            z[0] = 0
        else:
            target = self.target.index(self.csv[idx][1][0])
            y[int(target)] = 1
        if not self.logit:
            y = np.argmax(y)
        if not self.dicider:
            return x, y
        return x, z, y

    def __len__(self):
        if self.batch_size:
            return len(self.csv) // self.batch_size
        else:
            return len(self.csv)

    def __getitem__(self, idx):
        if not self.batch_size:
            return self.get_single_image(idx)


class TestGen:
    def __init__(self, rootpath, target_len, input_size=None, batch_size=None, normalize=lambda x: x):
        self.files = sorted(os.listdir(rootpath))
        self.rootpath = rootpath
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.batch_size = batch_size
        self.target_len = target_len
        self.normalize = normalize

    def get_single_image(self, idx):
        idx *= 4
        imname = self.files[idx].split('_')[0]
        x = np.zeros((*self.input_size, 4), dtype='float32')
        c = ['red', 'green', 'blue', 'yellow']
        for i in range(4):
            img = Image.open(os.path.join(self.rootpath, f'{imname}_{c[i]}.png')).resize(self.input_size)
            x[:, :, i] = np.array(img, dtype='float32')
        x = np.rollaxis(x, 2)
        x = self.normalize(x)
        return x

    def __len__(self):
        return len(self.files) // 4

    def __getitem__(self, idx):
        if not self.batch_size:
            return self.get_single_image(idx)
