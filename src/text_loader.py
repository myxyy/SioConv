import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Literal
from tqdm import tqdm
import os
import datasets

class TextDataset(Dataset):
    def __init__(self, path: str, size: int, tokenizer, transforms=None, tokenized_file_name=None) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms

        print('loading...')

        filename = path.split('/')[-1].split('.')[0]
        if tokenized_file_name is None:
            tokenized_text = None
        else:
            tokenized_text = np.load(tokenized_file_name) if os.path.isfile(tokenized_file_name) else None
        
        if tokenized_text is None:
            text_raw = open(path, 'r', encoding='utf-8').read()
            i = 0
            chunk_size = 2**26
            chunk_list = []
            print('tokenizing...')
            for i in tqdm(range(len(text_raw)//chunk_size+1)):
                chunk_list.append(tokenizer.encode(text_raw[i*chunk_size:(i+1)*chunk_size].lower()))
            print('merging...')
            self.text = np.array([token for chunk in chunk_list for token in chunk])
            np.save(tokenized_file_name, self.text)
        else:
            self.text = tokenized_text

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.text[index*self.size:(index+1)*self.size]
        data_next = self.text[index*self.size+1:(index+1)*self.size+1]

        if self.transforms is not None:
            data = self.transforms(data)
            data_next = self.transforms(data_next)
        return data, data_next


    def __len__(self) -> int:
        return (len(self.text)-1)//self.size

class HFDataset(Dataset):
    def __init__(self, dataset: datasets.arrow_dataset.Dataset, column: str, size: int, tokenizer, transforms=None, tokenized_file_name=None) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms


        if tokenized_file_name is None:
            tokenized_text = None
        else:
            print('loading...')
            tokenized_text = np.load(tokenized_file_name) if os.path.isfile(tokenized_file_name) else None
        
        if tokenized_text is None:
            bos_token = tokenizer.bos_token
            text_raw = bos_token.join(dataset[column])
            i = 0
            chunk_size = 2**26
            chunk_list = []
            print('tokenizing...')
            for i in tqdm(range(len(text_raw)//chunk_size+1)):
                chunk_list.append(tokenizer.encode(text_raw[i*chunk_size:(i+1)*chunk_size].lower()))
            print('merging...')
            self.text = np.array([token for chunk in chunk_list for token in chunk])
            np.save(tokenized_file_name, self.text)
        else:
            self.text = tokenized_text

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.text[index*self.size:(index+1)*self.size]
        data_next = self.text[index*self.size+1:(index+1)*self.size+1]

        if self.transforms is not None:
            data = self.transforms(data)
            data_next = self.transforms(data_next)
        return data, data_next


    def __len__(self) -> int:
        return (len(self.text)-1)//self.size