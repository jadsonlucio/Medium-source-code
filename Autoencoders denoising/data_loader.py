import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DataLoader():
    def __init__(self, train, test, val, img_res=(128, 128)):
        self.train = train
        self.test = test
        self.val = val
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        if data_type == "train":
            data = self.train
        elif data_type == "test":
            data = self.test

        images_idxs = np.random.choice(list(range(len(data))), size=batch_size)
        images = data[images_idxs]
        imgs_A = []
        imgs_B = []
        for img in images:

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = np.array(Image.fromarray(img_A).resize(self.img_res))
            img_B = np.array(Image.fromarray(img_B).resize(self.img_res))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"

        if data_type == "train":
            data = self.train
        elif data_type == "val":
            data = self.val

        self.n_batches = int(len(data) / batch_size)

        for i in range(self.n_batches-1):
            batch = data[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = np.array(Image.fromarray(img_A).resize(self.img_res))
                img_B = np.array(Image.fromarray(img_B).resize(self.img_res))
                
                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
