# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json

class VideoData(Dataset):
    def __init__(self, mode, dataset='SM_MrHiSum'):
        """ Custom Dataset class wrapper for loading the frame features, text features and ground truth importance scores.
        :param str mode: The mode of the model, train, val or test.
        :param str dataset: The name of the dataset: ['SM_VideoXum' | 'SM_MrHiSum']
        """
        self.mode = mode
        if dataset == 'SM_MrHiSum':
            self.filename = './dataset/sm_mrhisum.h5'
            self.dataset_split = './dataset/sm_mrhisum_split.json'
        else:
            raise ValueError("Error: no valid dataset. Must be: ['SM_VideoXum' | 'SM_MrHiSum']")

        # keep HDF5 filename, don't open yet
        self.hdf = h5py.File(self.filename, 'r')

        # load split info
        with open(self.dataset_split, 'r') as f:
            keys = json.load(f)

        # store only video names
        self.video_names = [v for v in self.hdf.keys() if v in keys[self.mode]]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # load data on demand
        with h5py.File(self.filename, 'r') as hdf:
            video_features = torch.from_numpy(hdf[video_name + '/video_embeddings'][()]).float()
            text_features = torch.from_numpy(hdf[video_name + '/qwen_text_embeddings'][()]).float()
            transcript_features = torch.from_numpy(hdf[video_name + '/aligned_transcripts'][()]).float()
            gtscore = torch.from_numpy(hdf[video_name + '/gtscores'][()]).float()
            gt_summary = torch.from_numpy(hdf[video_name + '/gt_summary'][()]).float()
            cps = hdf[video_name + '/change_points'][()]

        return video_features, text_features, transcript_features, gtscore, gt_summary, cps


def get_loader(mode, dataset='SM_MrHiSum'):
    """ Loads the dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str dataset: The name of the dataset
    :return The dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, dataset=dataset)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, dataset=dataset)


if __name__ == '__main__':
    pass