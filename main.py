from texts import CustomeTokenizer, TranslationInputData, TrainModel
from bid_transformer import Transformer

from datasets import load_dataset

from torch.utils.data import DataLoader, random_split
import torch

from pathlib import Path

def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair[lang]

def get_dataloaders() :

    # raw_data = load_dataset('cfilt/iitb-english-hindi', split='train')['translation'][:5000]
    raw_data = load_dataset("Helsinki-NLP/opus-100", "en-gu", split='train')['translation'][:5000]


    print(f"Size Of Data {len(raw_data)}")

    src_max_len = max([len(sen[config['src_lang']]) for sen in raw_data])
    tgt_max_len = max([len(sen[config['tgt_lang']]) for sen in raw_data])

    config['seq_len'] = max(src_max_len, tgt_max_len) + 2 # +2 for sos and eos

    tokenizer = CustomeTokenizer(config['tgt_lang']).build(get_all_sentences(raw_data, config['tgt_lang']))

    print(tokenizer.get_vocab_size())

    tokenizer2 = CustomeTokenizer(config['src_lang']).build(get_all_sentences(raw_data, config['src_lang']))

    tokenizer.add_tokens(list(tokenizer2.get_vocab()))

    train_data_size = int(.90 * len(raw_data))
    test_data_size = len(raw_data) - train_data_size

    print(f"Size Of Training Data {train_data_size}")
    print(f"Size Of Testing Data {test_data_size}")

    train_data, test_data = random_split(raw_data, [train_data_size, test_data_size])

    train_data = TranslationInputData(train_data, config['src_lang'], config['tgt_lang'], config['seq_len'], tokenizer)
    n_test_data = TranslationInputData(test_data, config['src_lang'], config['tgt_lang'], config['seq_len'], tokenizer)
    r_test_data = TranslationInputData(test_data, config['tgt_lang'], config['src_lang'], config['seq_len'], tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(n_test_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    r_test_dataloader = DataLoader(r_test_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    return (train_dataloader, test_dataloader, r_test_dataloader, tokenizer)

def start_training() : 

    train, test, r_test, tokenizer =  get_dataloaders()

    config['vocab_size']  = tokenizer.get_vocab_size()

    print(f"Vocab Size : {config['vocab_size']}")

    model = Transformer(config['dmodel'], config['vocab_size'], config['seq_len'], config['num_layers'], config['num_head'], config['dff'], config['dropout'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device : {device}')


    model.to(device)

    path = Path('checkpoint.pth')

    # if path.exists() :
    #     checkpoint = torch.load(path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = TrainModel()

    trainer.train(model, train, optimizer, loss_fn, config['epochs'], device)
    trainer.test(model, test, tokenizer, config['batch_size'], device)
    trainer.test(model, r_test, tokenizer, config['batch_size'], device)

    trainer.train(model, train, optimizer, loss_fn, config['epochs'], device)
    trainer.test(model, test, tokenizer, config['batch_size'], device)
    trainer.test(model, r_test, tokenizer, config['batch_size'], device)

    trainer.train(model, train, optimizer, loss_fn, config['epochs'], device)
    trainer.test(model, test, tokenizer, config['batch_size'], device)
    trainer.test(model, r_test, tokenizer, config['batch_size'], device)

    trainer.train(model, train, optimizer, loss_fn, config['epochs'], device)
    trainer.test(model, test, tokenizer, config['batch_size'], device)
    trainer.test(model, r_test, tokenizer, config['batch_size'], device)

# ------------------------------------------------------------- main ------------------------------------------------

config = {
    'src_lang'  : 'en',
    'tgt_lang'  : 'gu',
    'dmodel'    : 512,
    'vocab_size' : 0,
    'seq_len' : 0,
    'num_layers' : 3,
    'num_head' : 8,
    'dff' : 2048,
    'dropout' : 0.1,
    'lr' : 9.9**-4,
    'batch_size' : 10,
    'epochs' : 50
}

start_training()