import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile

class car(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(car, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform


        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root)
        assert os.path.isdir(os.path.join(self.root, 'car_ims'))
        assert os.path.isfile(os.path.join(self.root, 'car_nori.list'))
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        car_nori_path = os.path.join(self.root, 'car_nori.list')
        imgs_path = os.path.join(self.root, 'car_ims')

        car_nori = open(car_nori_path)

        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for annos in car_nori:
            cnt += 1
            if cnt < 3: continue
            annos_list = annos.split()

            image = Image.open(os.path.join(imgs_path, annos_list[1]))
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()
            if len(annos_list) == 8:
                label = int(annos_list[6]) - 1
                if annos_list[7] == '1':
                    train_cnt += 1
                    train_data.append(image_np)
                    train_labels.append(label)
                else:
                    test_cnt += 1
                    test_data.append(image_np)
                    test_labels.append(label)
            else:
                tmp = annos_list[5].split(')')
                label = int(tmp[1]) - 1
                if annos_list[6] == '1':
                    train_cnt += 1
                    train_data.append(image_np)
                    train_labels.append(label)
                else:
                    test_cnt += 1
                    test_data.append(image_np)
                    test_labels.append(label)
            if (cnt-2)%1000 == 0:
                print('{} images have been extracted'.format(cnt-2))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt-2, train_cnt, test_cnt))
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))