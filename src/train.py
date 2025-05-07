import torchvision
from text_loader import TextDataset
import torch
import torch.nn as nn
import os
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from torch.distributed.pipeline.sync import Pipe
import copy
from helper import calc_num_parameters
from torch.utils.tensorboard import SummaryWriter
import datetime
from transformers import PreTrainedTokenizer

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    transforms = torchvision.transforms.Compose([])
    tokenizer: PreTrainedTokenizer = instantiate(cfg.tokenizer)
    vocab_size = tokenizer.vocab_size
    dataset = instantiate(cfg.train.dataset)
    ckpt_path = cfg.train.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    devices = cfg.train.devices

    print('loading model...')

    model = instantiate(cfg.model)
    model = model(devices=devices, vocab_size=vocab_size, out_only_device=cfg.train.out_only_device)
    model.to(instantiate(cfg.train.dtype))

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        if cfg.train.reset_steps:
            epochs = 0
            steps = 0
            log_dir = cfg.train.log_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            epochs = ckpt['epochs']
            steps = ckpt['steps']
            log_dir = ckpt['log_dir']
        optimizer = instantiate(cfg.train.optimizer)
        optimizer = optimizer(params=model.parameters())
        if not cfg.train.reset_steps:
            optimizer.load_state_dict(ckpt['optimizer'])

        if cfg.train.scheduler is None:
            scheduler = None
        else:
            scheduler = instantiate(cfg.train.scheduler)
            scheduler = scheduler(optimizer=optimizer)
            if not cfg.train.reset_steps:
                scheduler.load_state_dict(ckpt['scheduler'])

        if not cfg.train.reset_steps:
            model.set_hidden(ckpt['hidden'])
        del ckpt
    else:
        epochs = 0
        steps = 0
        optimizer = instantiate(cfg.train.optimizer)
        optimizer = optimizer(params=model.parameters())
        log_dir = cfg.train.log_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if cfg.train.scheduler is None:
            scheduler = None
        else:
            scheduler = instantiate(cfg.train.scheduler)
            scheduler = scheduler(optimizer=optimizer)

    logger: SummaryWriter = instantiate(cfg.train.logger)
    logger = logger(log_dir=log_dir)

    total_steps = len(dataset) // cfg.train.batch_size_per_acc
    print(f'loaded. steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')

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
    if scheduler is not None:
        backup_scheduler_state_dict = copy.deepcopy(find_tensor_and_transfer(scheduler.state_dict()))
    backup_hidden = copy.deepcopy(find_tensor_and_transfer(model.get_hidden()))

    def save():
        print(f'saving... steps:{steps}/{total_steps} epochs:{epochs}/{cfg.train.max_epochs}')
        torch.save({
            'model': model.state_dict(),
            'steps': steps,
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'hidden': model.get_hidden(),
            'log_dir': log_dir,
        }, cfg.train.weight)

    def save_backup():
        print(f'saving... steps:{backup_steps}/{total_steps} epochs:{backup_epochs}/{cfg.train.max_epochs}')
        torch.save({
            'model': backup_model_state_dict,
            'steps': backup_steps,
            'epochs': backup_epochs,
            'optimizer': backup_optimizer_state_dict,
            'scheduler': backup_scheduler_state_dict if scheduler is not None else None,
            'hidden': backup_hidden,
            'log_dir': log_dir,
        }, cfg.train.weight)

    model.set_is_refresh(cfg.train.is_refresh)

    last_steps = steps

    try:
        for _ in range(cfg.train.max_epochs - epochs):
            pbar = tqdm(range(total_steps-steps), initial=steps, total=total_steps)
            loss_sum = 0
            previous_loss_sum = None
            num_tokens = 0
            for _ in pbar:
                if steps > last_steps and steps % cfg.train.save_every_n_steps == 0:
                    save()
                    #last_steps = steps
                if steps % cfg.train.backup_every_n_steps == 0:
                    #print('backup...')
                    backup_model_state_dict = copy.deepcopy(find_tensor_and_transfer(model.state_dict()))
                    backup_steps = steps
                    backup_epochs = epochs
                    backup_optimizer_state_dict = copy.deepcopy(find_tensor_and_transfer(optimizer.state_dict()))
                    if scheduler is not None:
                        backup_scheduler_state_dict = copy.deepcopy(find_tensor_and_transfer(scheduler.state_dict()))
                    backup_hidden = copy.deepcopy(find_tensor_and_transfer(model.get_hidden()))

                if cfg.train.refresh_every_n_steps is not None and steps % cfg.train.refresh_every_n_steps == 0:
                    model.reset_hidden()
                
                if steps % cfg.train.num_acc == 0 and steps > last_steps:
                    if (steps // cfg.train.num_acc) % cfg.train.log_every_n_updates == 0:
                        logger.add_scalar("loss", loss_sum, steps)
                        logger.add_scalar("lr", optimizer.param_groups[0]["lr"], steps)
                    previous_loss_sum = loss_sum
                    loss_sum = 0
                    if cfg.train.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                if steps % cfg.train.num_acc == 0:
                    _, text_next_acc = dataset[cfg.train.batch_size_per_acc * steps:cfg.train.batch_size_per_acc * (steps+cfg.train.num_acc)]
                    num_tokens = (text_next_acc != tokenizer.pad_token_id).sum()

                text, text_next = dataset[cfg.train.batch_size_per_acc * steps:cfg.train.batch_size_per_acc * (steps+1)]
                #print(text)
                #print(text_next)
                text = torch.from_numpy(text).to(devices[0])
                text_next = torch.from_numpy(text_next).to(devices[-1])
                text = text.long()

                text_hat = model_pipe(text).local_value()

                loss = nn.functional.cross_entropy(text_hat.view(-1,vocab_size), text_next.view(-1).long(), ignore_index=tokenizer.pad_token_id, reduction="sum")
 
                loss_norm = loss / num_tokens
                loss_norm.backward()
                loss_sum += loss_norm

                pbar.set_postfix({"loss": "-" if previous_loss_sum is None else previous_loss_sum.item(), "loss_temp":loss_norm.item() * cfg.train.num_acc, "lr":optimizer.param_groups[0]["lr"]})
                steps += 1
            steps = 0
            epochs += 1
            save()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupted')
        if steps - last_steps > cfg.train.backup_every_n_steps:
            save_backup()


if __name__ == '__main__':
    main()