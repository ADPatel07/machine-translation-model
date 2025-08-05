from torch.utils.data import Dataset,DataLoader, random_split
import torch
import torch.nn.functional as F
import math

from pathlib import Path
from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import WordLevel
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

class CustomeTokenizer() : 
    def __init__(self, lang, path='tokenizer_{0}.json') :
        super().__init__()
        self.tokenizer_path = Path(path.format(lang))
        self.lang = lang
        self.tokenizer = None


    def build(self, ds): # language detection
        if not self.tokenizer_path.exists():
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
            tokenizer.train_from_iterator(ds, trainer=trainer)
            tokenizer.save(str(self.tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        self.tokenizer = tokenizer
        return tokenizer

def _mask(size) :
  return torch.triu(torch.ones(size, size), diagonal=1).bool().unsqueeze(0)

# ---------------------------- String to Token Dataset ---------------------------------------------------

class TranslationInputData(Dataset) :
    def __init__(self, ds, src_lang, tgt_lang, seq_len, tokenizer) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([self.tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) :
        return len(self.ds)

    def __getitem__(self, idx:int) :
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # calculating how many padding token need
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #for sos and eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        #Input For Encoder
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        #Input For Decoder
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        #Expected Output
        output_tokens = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        return {'encoder_input' : encoder_input,
                'decoder_input' : decoder_input,
                'encoder_mask'  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                'decoder_mask'  : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & _mask(decoder_input.size(0)),
                'src_text'      : src_text,
                'tgt_text'      : tgt_text,
                'output_tokens' : output_tokens}


def weight_schedule(epoch, total_epochs) : 
    return min(1.0, epoch / (total_epochs * .5))

class TrainModel : 
  
    def train(self, model, train_dataloader, optimizer, loss_fn, epochs, device) :

        model.to(device)
        model.loss_fn = loss_fn

        num_classes = model.vocab_size
        max_loss = torch.log(torch.tensor(num_classes, dtype=torch.float)).item()

        model.train()
        scaler = GradScaler()

        for epoch in range(epochs) :

            batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')

            epoch_loss = 0
            num_samples = 0

            epoch_bai_loss = 0

            for batch in batch_iterator :
                optimizer.zero_grad()
            
                with autocast(False):
                    encoder_input = batch['encoder_input'].to(device)
                    decoder_input = batch['decoder_input'].to(device)
                    encoder_mask = batch['encoder_mask'].to(device)
                    decoder_mask = batch['decoder_mask'].to(device)

                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    output = model.projection(decoder_output)

                    loss = loss_fn(output.view(-1, model.vocab_size), batch['output_tokens'].to(device).view(-1))

                    # -------------------------------- BID ----------------------------------------------------------
                    E = encoder_output.clone().detach().requires_grad_(False) #pivots
                    D = model.embedding(decoder_input)
                    H = D.size(-1)
                    M = D.size(-2)

                    similarity = torch.bmm(D, E.transpose(-2, -1)) / math.sqrt(H)
                    # similarity = (D @ E.transpose(-1, -2)) / math.sqrt(H)
                    R = torch.bmm(F.softmax(similarity, dim=-1), E)

                    bai_loss = F.mse_loss(R, D)

                    lambda_value = weight_schedule(epoch, epochs)
                    # lambda_value = 10**-3

                    total_loss = loss + lambda_value * bai_loss
                    epoch_loss += total_loss.item() * encoder_input.size(0)
                    epoch_bai_loss += bai_loss.item() * encoder_input.size(0)
                    num_samples += encoder_input.size(0)

                    loss_percentage = (total_loss.item() / max_loss) * 100


                batch_iterator.set_postfix({
                        "loss": f"{total_loss.item():6.3f} ({loss_percentage:6.2f}%)", 
                        "Avg loss": f"{(epoch_loss / num_samples):6.3f} ({(epoch_loss / (num_samples * max_loss)) * 100:6.2f}%)",
                        "Avg bai_loss" : f"{epoch_bai_loss / num_samples : 6.3f}"
                    })

               
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': total_loss
                        }
                
            torch.save(checkpoint, 'checkpoint.pth')
            

    def test(self, model, test_dataloader, tokenizer, batch_size, device) :
        model.eval()

        count = 0

        with torch.no_grad() :
            for batch in test_dataloader :

                count += 1

                src = batch['encoder_input'].to(device)
                src_mask = batch['encoder_mask'].to(device)

                model_output = model.getGreedyDecode(src, src_mask, device, batch_size, tokenizer.token_to_id("[SOS]"), tokenizer.token_to_id("[EOS]"))

                output_text = tokenizer.decode(model_output[0].detach().cpu().numpy())

                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]

                print('-'*100)
                print('-'*100)
                print(f'SOURCE: {source_text}')
                print(f'TARGET: {target_text}')
                print(f'PREDICTED: {output_text}')

                if count == 2 :
                    break