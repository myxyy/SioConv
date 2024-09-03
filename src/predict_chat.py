import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import hydra
from hydra.utils import instantiate
from helper import calc_num_parameters

np.set_printoptions(threshold=np.inf)

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    devices = cfg.predict.devices
    ckpt = torch.load(cfg.predict.weight)
    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model)
    vocab_size = tokenizer.vocab_size
    model = model(devices=devices, vocab_size=vocab_size)
    model.to(instantiate(cfg.predict.dtype))
    model.load_state_dict(ckpt['model'])
    hidden_init = ckpt['hidden']
    model.eval()
    context_len = cfg.predict.context_len
    out_length = cfg.predict.max_len
    temperature = cfg.predict.temperature

    del ckpt
    torch.cuda.empty_cache()

    num_parameters = calc_num_parameters(model)
    print(f"#parameter:{num_parameters}")

    model.reset_hidden()

    def topp(x, p):
        xsv, xsi = torch.sort(x, descending=True)
        i_to_remove = torch.cumsum(xsv, dim=-1) > p
        i_to_remove[..., 1:] = i_to_remove[..., :-1].clone()
        i_to_remove[..., 0] = False
        xsv[i_to_remove] = 0
        return torch.gather(xsv, -1, xsi.argsort(-1))

    def predict(prompt):
        prompt = torch.from_numpy(np.array(tokenizer.encode(prompt)).astype(int)).clone().to(devices[0])
        prompt_len = len(prompt)
        prompt = torch.nn.functional.pad(prompt, (0, out_length-prompt_len), 'constant', 0)

        beam_width = 1
        #model.set_hidden(hidden_init)

        current_len = 0
        model.set_is_refresh(True)
        prompt_beam = prompt.repeat(beam_width, 1)
        while current_len <= prompt_len:
            x = prompt_beam[:,current_len:current_len+context_len]
            x = x.long()
            if (prompt_len - current_len <= context_len):
                model.set_is_refresh(prompt_len - current_len == context_len)
                predict_init = model(x) # (1, context_len, vocab_size)
                #predict_init_i = predict_init.view(context_len, vocab_size)[prompt_len - current_len -1].topk(beam_width)
                predict_init_i = torch.multinomial(topp(nn.Softmax(dim=1)(predict_init[:,prompt_len-current_len-1,:]/temperature), cfg.predict.top_p), 1)
                prompt_beam[:,prompt_len] = predict_init_i
                current_len = prompt_len + 1
            else:
                model.set_is_refresh(True)
                model(x)
                current_len += context_len

        out_last = 0

        while current_len < out_length:
            model.set_is_refresh(current_len % context_len == 0)
            complete_len = ((current_len-1) // context_len) * context_len
            x = prompt_beam[:,complete_len:complete_len+context_len]
            x = x.long()
            predict_beam = model(x).to(devices[0])
            predict_beam_i = torch.multinomial(topp(nn.Softmax(dim=1)(predict_beam[:,current_len-complete_len-1,:]/temperature), cfg.predict.top_p), 1)
            prompt_beam[:,current_len] = predict_beam_i

            current_len += 1
            if predict_beam_i == tokenizer.bos_token_id:
                break

            predict = prompt_beam[0]
            predict = predict.cpu().numpy()

            chars = tokenizer.decode(predict[prompt_len:current_len].tolist())
            #print(predict[0:current_len+10])
            #print(chars)
            delimiter_ind = max(chars[out_last:].find(" "), chars[out_last:].find("\n")) + out_last
            if delimiter_ind >= out_last:
                print(chars[out_last:delimiter_ind+1], end='', flush=True)
                out_last = delimiter_ind+1
            elif len(chars[out_last:]) > 1:
                print(chars[out_last:out_last+1], end='', flush=True)
                out_last += 1

        chars = tokenizer.decode(predict[prompt_len:current_len-1].tolist(), clean_up_tokenization_spaces=True)[out_last:]
        print(chars, end='', flush=True)


        predict = prompt_beam[0]
        predict = predict.cpu().numpy()
        predict = tokenizer.decode(predict.tolist())
        return predict

    while True:
        prompt = input('Question:\n')
        if prompt == "reset":
            model.reset_hidden()
            print("Reset hidden state\n")
        else:
            print('\nAnswer:')
            predict(prompt + tokenizer.bos_token)
            print('\n')

if __name__ == '__main__':
    main()