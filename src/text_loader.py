import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Literal
from tqdm import tqdm
import os
import datasets

class TextDataset(Dataset):
    def __init__(self, path: str, size: int, tokenizer, tokenized_file_name=None) -> None:
        super().__init__()
        self.size = size

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
                chunk_list.append(tokenizer.encode(text_raw[i*chunk_size:(i+1)*chunk_size].lower(), add_special_tokens=False))
            print('merging...')
            text = np.array([token for chunk in chunk_list for token in chunk])
            np.save(tokenized_file_name, text)
        else:
            text = tokenized_text
        text_num = (len(text)-1)//self.size
        self.text_num = text_num
        self.text = text[0:text_num * self.size].reshape(text_num, self.size)
        self.text_next = text[1:text_num * self.size+1].reshape(text_num, self.size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text[index], self.text_next[index]


    def __len__(self) -> int:
        return self.text_num

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
                chunk_list.append(tokenizer.encode(text_raw[i*chunk_size:(i+1)*chunk_size].lower(), add_special_tokens=False))
            print('merging...')
            text = np.array([token for chunk in chunk_list for token in chunk])
            np.save(tokenized_file_name, text)
        else:
            text = tokenized_text
        text_num = (len(text)-1)//self.size
        self.text_num = text_num
        self.text = text[0:text_num * self.size].reshape(text_num, self.size)
        self.text_next = text[1:text_num * self.size+1].reshape(text_num, self.size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text[index], self.text_next[index]

    def __len__(self) -> int:
        return self.text_num

class InterleaveHFDataset(Dataset):
    def __init__(self, dataset_list, column_list, tokenizer, size, cache_dir=None, seed=0):
        self.tokenizer = tokenizer
        self.size = size
        super().__init__()
        if cache_dir is not None and os.path.isdir(cache_dir):
            self.dataset = datasets.load_from_disk(cache_dir)
        else:
            normalized_dataset_list = []
            probabilities = []
            len_sum = 0
            for dataset in dataset_list:
                len_sum += len(dataset)
            for i, dataset in enumerate(dataset_list):
                dataset.remove_columns([column for column in dataset.column_names if column != column_list[i]])
                if "text" not in dataset.column_names:
                    dataset.rename_column(column_list[i], "text")
                normalized_dataset_list.append(dataset)
                probabilities.append(len(dataset) / len_sum)
            self.dataset = datasets.interleave_datasets(normalized_dataset_list, probabilities=probabilities, stopping_strategy="all_exhausted", seed=seed)
            if cache_dir is not None:
                self.dataset.save_to_disk(cache_dir)

    def __getitem__(self, index: int):
        tokenized_data = self.tokenizer(self.dataset[index]["text"], padding="max_length", max_length=self.size+1, truncation=True)
        input_ids = np.array(tokenized_data["input_ids"])
        #attention_mask = np.array(tokenized_data["attention_mask"])
        return input_ids[...,:self.size], input_ids[...,1:]

    def __len__(self):
        return len(self.dataset)