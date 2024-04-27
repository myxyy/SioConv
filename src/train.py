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
from helper import calc_num_parameters

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
        model = instantiate(ckpt['model_config'])
        model = model(devices=devices, vocab_size=vocab_size, out_only_device=cfg.train.out_only_device)
        model.load_state_dict(ckpt['model'])
        epochs = ckpt['epochs']
        steps = ckpt['steps']
        optimizer = instantiate(ckpt['optimizer_config'])
        optimizer = optimizer(params=model.parameters())
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler = instantiate(ckpt['scheduler_config'])
        scheduler = scheduler(optimizer=optimizer)
        scheduler.load_state_dict(ckpt['scheduler'])
        del ckpt
    else:
        model = instantiate(cfg.model)
        model = model(devices=devices, vocab_size=vocab_size, out_only_device=cfg.train.out_only_device)
        epochs = 0
        steps = 0
        optimizer = instantiate(cfg.train.optimizer)
        optimizer = optimizer(params=model.parameters())
        scheduler = instantiate(cfg.train.scheduler)
        scheduler = scheduler(optimizer=optimizer)

    total_steps = len(dataset) // cfg.train.batch_size_per_acc
    print(f'loaded. steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')

    dtype = model.dtype

    torch.cuda.empty_cache()

    num_parameters = calc_num_parameters(model)
    print(f"#parameter:{num_parameters}")

    model_pipe = nn.Sequential(*model.module_list())
    model_pipe = Pipe(model_pipe, chunks=cfg.train.batch_size_per_acc, checkpoint=cfg.train.pipeline_checkpoint)
    model_pipe.train()

    def find_tensor_and_transfer(d):
        return {k: v.cpu() if isinstance(v, torch.Tensor) else find_tensor_and_transfer(v) for k, v in d.items()} if isinstance(d, dict) else d

    backup_model_state_dict = copy.deepcopy(find_tensor_and_transfer(model.state_dict()))
    backup_steps = steps
    backup_epochs = epochs
    backup_optimizer_state_dict = copy.deepcopy(find_tensor_and_transfer(optimizer.state_dict()))
    backup_scheduler_state_dict = copy.deepcopy(find_tensor_and_transfer(scheduler.state_dict()))

    def save():
        print(f'saving... steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')
        torch.save({
            'model': model.state_dict(),
            'steps': steps,
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model_config': cfg.model,
            'optimizer_config': cfg.train.optimizer,
            'scheduler_config': cfg.train.scheduler,
        }, cfg.train.weight)

    def save_backup():
        print(f'saving... steps:{backup_steps}/{total_steps} epochs:{backup_epochs}/{cfg.train.max_epochs}')
        torch.save({
            'model': backup_model_state_dict,
            'steps': backup_steps,
            'epochs': backup_epochs,
            'optimizer': backup_optimizer_state_dict,
            'scheduler': backup_scheduler_state_dict,
            'model_config': cfg.model,
            'optimizer_config': cfg.train.optimizer,
            'scheduler_config': cfg.train.scheduler,
        }, cfg.train.weight)

    model.set_is_refresh(True)

    last_steps = steps

    try:
        for _ in range(cfg.train.max_epochs - epochs):

            dataloader = torch.utils.data.DataLoader([dataset[i] for i in range(cfg.train.batch_size_per_acc * steps, len(dataset))], batch_size=cfg.train.batch_size_per_acc, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)

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
                    backup_scheduler_state_dict = copy.deepcopy(find_tensor_and_transfer(scheduler.state_dict()))

                if steps % cfg.train.refresh_every_n_steps == 0:
                    model.reset_hidden()

                text, text_next = batch
                text = text.to(devices[0])
                text_next = text_next.to(devices[-1])
                text = text.long()

                text_hat = model_pipe(text).local_value()

                loss = nn.CrossEntropyLoss()(text_hat.view(-1,vocab_size), text_next.view(-1).long())
 
                loss_norm_acc = loss / cfg.train.num_acc
                loss_norm_acc.backward()

                if steps % cfg.train.num_acc == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                pbar.set_postfix({"loss":loss.item(), "lr":optimizer.param_groups[0]["lr"]})
                steps += 1
            steps = 0
            epochs += 1
            save()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupted')
        if steps - last_steps > cfg.train.backup_every_n_steps == 0:
            save_backup()


if __name__ == '__main__':
    main()