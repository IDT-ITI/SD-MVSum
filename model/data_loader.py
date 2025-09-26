# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json


class VideoData(Dataset):
    def __init__(self, mode, dataset='S_VideoXum'):
        self.mode = mode
        self.dataset = dataset

        if dataset == 'S_VideoXum':
            self.filename = './dataset/script_videoxum.h5'
            self.dataset_split = './dataset/script_videoxum_split.json'
        elif dataset == 'S_MrhiSum':
            self.filename = './dataset/final_mrhisum.h5'
            self.dataset_split = './dataset/filtered_mr_hisum_split.json'
        else:
            raise ValueError("Invalid dataset.")

        self.hdf = h5py.File(self.filename, 'r')
        with open(self.dataset_split, 'r') as f:
            keys = json.load(f)

        self.video_names = [v for v in self.hdf.keys() if v in keys[self.mode]]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]

        with h5py.File(self.filename, 'r') as hdf:
            video_features = torch.from_numpy(hdf[video_name + '/video_embeddings'][()]).float()
            text_features = torch.from_numpy(hdf[video_name + '/text_embeddings'][()]).float()
            gtscore = torch.from_numpy(hdf[video_name + '/gtscores'][()]).float()
            transcripts = torch.from_numpy(hdf[video_name + '/aligned_transcripts'][()]).float()

            if self.dataset == 'S_MrhiSum':
                cps = torch.from_numpy(hdf[video_name + '/change_points'][()]).float()
                gt_summary = torch.from_numpy(hdf[video_name + '/gt_summary'][()]).float()
                return video_features, text_features, transcripts, gtscore, cps, gt_summary
            else:
                return video_features, text_features, transcripts, gtscore



def get_loader(mode, dataset='S_VideoXum'):
    """ Loads the dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str dataset: The name of the dataset: ['S_VideoXum' | 'S_NewsVSum']
    :return: The dataset used in each mode.
    """
    vd = VideoData(mode, dataset=dataset)

    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return DataLoader(vd, batch_size=1, shuffle=False)


if __name__ == '__main__':
    pass