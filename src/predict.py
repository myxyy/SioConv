import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import hydra
from hydra.utils import instantiate

np.set_printoptions(threshold=np.inf)

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    devices = cfg.predict.devices
    ckpt = torch.load(cfg.predict.weight)
    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(ckpt['model'])
    vocab_size = tokenizer.vocab_size
    model = model(devices=devices, vocab_size=vocab_size)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    context_len = cfg.predict.context_len
    out_length = cfg.predict.max_len
    dtype = model.dtype
    temperature = cfg.predict.temperature

    del ckpt
    torch.cuda.empty_cache()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#parameter:{num_parameters}")

    def predict(prompt):
        prompt = torch.from_numpy(np.array(tokenizer.encode(prompt)).astype(int)).clone().to(devices[0])
        prompt_len = len(prompt)
        prompt = torch.nn.functional.pad(prompt, (0, out_length-prompt_len), 'constant', 0)

        beam_width = 1
        model.reset_hidden()

        current_len = 0
        start = 0
        model.set_is_refresh(True)
        prompt_beam = prompt.repeat(beam_width, 1)
        while current_len < prompt_len:
            x = prompt_beam[:,current_len:current_len+context_len]
            x = x.long()
            if (prompt_len - current_len <= context_len):
                model.set_is_refresh(False)
                predict_init = model(x) # (1, context_len, vocab_size)
                #predict_init_i = predict_init.view(context_len, vocab_size)[prompt_len - current_len -1].topk(beam_width)
                predict_init_i = torch.multinomial(nn.Softmax(dim=1)(predict_init[:,prompt_len-current_len-1,:]/temperature), 1)
                prompt_beam[:,prompt_len] = predict_init_i
                current_len = prompt_len
            else:
                model.set_is_refresh(True)
                model(x)
                current_len += context_len
                start += context_len

        out_last = 0

        while current_len < out_length:
            model.set_is_refresh(current_len % context_len == 0)
            x = prompt_beam[:,start:start+context_len]
            x = x.long()
            predict_beam = model(x).to(devices[0])
            #_, predict_beam_i = predict_beam[:,current_len-1-start,:].reshape(beam_width * vocab_size).topk(beam_width)
            predict_beam_i = torch.multinomial(nn.Softmax(dim=1)(predict_beam[:,current_len-1-start,:]/temperature), 1)
            #prompt_beam = prompt_beam[torch.div(predict_beam_i, vocab_size, rounding_mode='floor')]
            #prompt_beam[:,current_len] = predict_beam_i % vocab_size 
            prompt_beam[:,current_len] = predict_beam_i

            predict = prompt_beam[0]
            predict = predict.cpu().numpy()
            chars = tokenizer.decode(predict[out_last:current_len+1].tolist())
            

            if '\ufffd' not in chars:
                print(chars, end='', flush=True)
                out_last = current_len + 1
            elif current_len - out_last > 16:
                is_break = False
                out_last_skip_error = out_last
                while out_last_skip_error < current_len:
                    out_last_next = out_last_skip_error + 1
                    while out_last_next < current_len:
                        chars = tokenizer.decode(predict[out_last_skip_error:out_last_next].tolist())
                        if '\ufffd' not in chars:
                            chars = tokenizer.decode(predict[out_last:out_last_next].tolist())
                            print(chars, end='', flush=True)
                            is_break = True
                            out_last = out_last_next
                            break
                        else:
                            out_last_next += 1
                    if is_break:
                        break
                    else:
                        out_last_skip_error += 1
                        


            current_len += 1

            if current_len % context_len == 1 or context_len == 1:
                start = start + context_len

        chars = tokenizer.decode(predict[out_last:].tolist())
        print(chars, end='', flush=True)


        predict = prompt_beam[0]
        predict = predict.cpu().numpy()
        predict = tokenizer.decode(predict.tolist())
        return predict

    while True:
        prompt = input('prompt:')
        predict(prompt)
        print('\n')

if __name__ == '__main__':
    main()