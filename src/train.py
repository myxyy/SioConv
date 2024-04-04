import torchvision
from text_loader import TextDataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from torch.distributed.pipeline.sync import Pipe
import copy

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    transforms = torchvision.transforms.Compose([])
    tokenizer = instantiate(cfg.tokenizer)
    vocab_size = tokenizer.vocab_size
    dataset = TextDataset(cfg.train.text, cfg.train.length, tokenizer, transforms, tokenized_text_dir_path=cfg.tokenized_text_dir_path)
    ckpt_path = cfg.train.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    #trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_every_n_steps, logger=[TensorBoardLogger('./')])
    devices = cfg.train.devices

    print('loading model...')

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model = instantiate(ckpt['model'])
        model = model(devices=devices, vocab_size=vocab_size)
        model.load_state_dict(ckpt['state_dict'])
        epochs = ckpt['epochs']
        steps = ckpt['steps']
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
        optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
    else:
        model = instantiate(cfg.model)
        model = model(devices=devices, vocab_size=vocab_size)
        epochs = 0
        steps = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    total_steps = len(dataset) // cfg.train.batch_size
    print(f'loaded. steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')

    dtype = model.dtype

    torch.cuda.empty_cache()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#parameter:{num_parameters}")

    model_pipe = nn.Sequential(*model.module_list())
    model_pipe = Pipe(model_pipe, chunks=cfg.train.batch_size, checkpoint='except_last')
    model_pipe.train()

    def find_tensor_and_transfer(d):
        return {k: v.cpu() if isinstance(v, torch.Tensor) else find_tensor_and_transfer(v) for k, v in d.items()} if isinstance(d, dict) else d

    backup_model_state_dict = copy.deepcopy(find_tensor_and_transfer(model.state_dict()))
    backup_steps = steps
    backup_epochs = epochs
    backup_optimizer_state_dict = copy.deepcopy(find_tensor_and_transfer(optimizer.state_dict()))

    def save():
        print(f'saving... steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')
        torch.save({
            'state_dict': model.state_dict(),
            'steps': steps,
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'model': cfg.model,
        }, cfg.train.weight)

    def save_backup():
        print(f'saving... steps:{backup_steps}/{total_steps} epochs:{backup_epochs}/{cfg.train.max_epochs}')
        torch.save({
            'state_dict': backup_model_state_dict,
            'steps': backup_steps,
            'epochs': backup_epochs,
            'optimizer': backup_optimizer_state_dict,
            'model': cfg.model,
        }, cfg.train.weight)

    model.set_is_refresh(True)

    try:
        for _ in range(cfg.train.max_epochs - epochs):

            dataloader = torch.utils.data.DataLoader([dataset[i] for i in range(cfg.train.batch_size * steps, len(dataset))], batch_size=cfg.train.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)

            pbar = tqdm(dataloader, initial=steps, total=total_steps)
            for batch in pbar:
                if steps > 0 and steps % cfg.train.save_every_n_steps == 0:
                    save()
                if steps % cfg.train.backup_every_n_steps == 0:
                    #print('backup...')
                    backup_model_state_dict = copy.deepcopy(find_tensor_and_transfer(model.state_dict()))
                    backup_steps = steps
                    backup_epochs = epochs
                    backup_optimizer_state_dict = copy.deepcopy(find_tensor_and_transfer(optimizer.state_dict()))

                model.reset_hidden()
                optimizer.zero_grad()

                text, text_next = batch
                text = text.to(devices[0])
                text_next = text_next.to(devices[-1])
                text = text.long()

                text_hat = model_pipe(text).local_value()

                loss = nn.CrossEntropyLoss()(text_hat.view(-1,vocab_size), text_next.view(-1).long())
 
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                steps += 1
            steps = 0
            epochs += 1
            save()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupted')
        save_backup()


if __name__ == '__main__':
    main()