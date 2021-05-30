import scipy.io as scio
import os
import numpy as np
import torch

class ParentDataset:
    def __init__(self, data_features_dir):
        """
        {'__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Mon May  8 16:33:27 2017',
        '__version__': '1.0',
        '__globals__': [],
        'features': array([[1.27028406e-01, 0.00000000e+00, 3.83341342e-01, ...,
                1.67552959e-02, 1.21401340e-01, 2.43460596e-01],
                ...,
                [4.07616198e-01, 2.56229609e-01, 3.31381522e-02, ...,
                3.53541344e-01, 2.25771200e-02, 6.21905439e-02]]),
        'image_files': array([[array(['/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/antelope/antelope_10001.jpg'],
            dtype='<U83')],
                [array(['/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/antelope/antelope_10002.jpg'],
            dtype='<U83')],
                ...,
                [array(['/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/zebra/zebra_11170.jpg'],
            dtype='<U77')]], dtype=object),
        'labels': array([[ 1],
                [ 1],
                ...,
                [38]], dtype=uint8)}
        """
        res101_data = scio.loadmat(os.path.join(data_features_dir, 'res101.mat'))

        # TODO: consider normalization
        self.features = res101_data['features'].T
        self.labels = res101_data['labels'].reshape(-1) - 1
        self.image_files = []
        for i in range(res101_data['image_files'].shape[0]):
            path = res101_data['image_files'][i][0][0]
            x, filename = os.path.split(path)
            _, classdir = os.path.split(x)
            self.image_files.append(os.path.join(classdir, filename))   # E.g.: "antelope/antelope_10002.jpg"
        assert self.labels.min() == 0
        assert self.labels.shape[0] == self.features.shape[0]
        assert self.labels.shape[0] == len(self.image_files)

        """
        {'__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Fri Aug 21 10:36:20 2020',
        '__version__': '1.0',
        '__globals__': [],
        'allclasses_names': array([[array(['antelope'], dtype='<U8')],
                [array(['grizzly+bear'], dtype='<U12')],
                ...
                [array(['dolphin'], dtype='<U7')]], dtype=object),
        'att': array([[-0.00375358,  0.12045618,  0.26584459, ...,  0.22516498,
                0.19613947,  0.03819588],
                ...
                [ 0.03145501,  0.03495531,  0.04915256, ...,  0.01771   ,
                0.25883601,  0.14194515]]),
        'original_att': array([[-1.  , 39.25, 83.4 , ..., 63.57, 55.31, 10.22],
                ...
                [ 8.38, 11.39, 15.42, ...,  5.  , 72.99, 37.98]]),
        'test_seen_loc': array([[11061],
                [ 6271],
                ...,
                [33202]], dtype=uint16),
        'test_unseen_loc': array([[ 1047],
                [ 1048],
                ...,
                [35291]], dtype=uint16),
        'train_loc': array([[21483],
                [11453],
                ...,
                [ 5161]], dtype=uint16),
        'trainval_loc': array([[21483],
                [11453],
                ...,
                [ 5161]], dtype=uint16),
        'val_loc': array([[12751],
                [15468],
                ...,
                [ 7993]], dtype=uint16)}
        """
        attribute_dict = scio.loadmat(os.path.join(data_features_dir, 'att_splits.mat'))

        self._attribute_dict = attribute_dict
        self.attributes = attribute_dict['att'].T

        self.class_name_to_idx = {}
        self.class_idx_to_name = []
        for i in range(attribute_dict['allclasses_names'].shape[0]):
            class_name = attribute_dict['allclasses_names'][i][0][0]
            self.class_name_to_idx[class_name] = i
            self.class_idx_to_name.append(class_name)

        self.class_num = len(self.class_name_to_idx)

        assert self.attributes.shape[0] == len(self.class_name_to_idx)

    def get_dataset(self, type, device):
        """
        :param type: name of the data set to get: 'train', 'val', 'trainval', 'test_seen', 'test_unseen'.
        :returns: (features, labels, mask) tuple, where:
            * features - input features matrix of shape (N, D)
            * labels - labels of shape (N,), where each element is a number from 0 to C-1.
            * mask - mask vector with 1 where legal classes of this dataset are, of shape (C,)
        """
        features = torch.FloatTensor(self.features[self._attribute_dict[f'{type}_loc'].reshape(-1) - 1, :]).to(device)
        labels = torch.LongTensor(self.labels[self._attribute_dict[f'{type}_loc'].reshape(-1) - 1]).to(device)
        classes = torch.unique(labels)
        mask = torch.zeros((self.class_num,), dtype=torch.bool).to(device)
        for label in classes:
            mask[label.item()] = True
        return features, labels, classes, mask
    

"""
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
"""