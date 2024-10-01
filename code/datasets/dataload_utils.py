from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import os
import pandas as pd
from preprocess import preprocess

cur_dir = Path(os.getcwd())

train = preprocess(pd.read_csv(cur_dir.parent.parent/'data/processed/train.csv'))
val = preprocess(pd.read_csv(cur_dir.parent.parent/'data/processed/val.csv'))


def yield_tokens(df):
    for _, sample in df.iterrows():
        yield sample.to_list()[2]


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

vocab = build_vocab_from_iterator(yield_tokens(train), specials=special_symbols)
vocab.set_default_index(UNK_IDX)

sample = train['Text'][2]
print(f"Sample: {sample}")
print(f"Type: {type(sample)}")
encoded = vocab(sample)
print(encoded)

torch.manual_seed(420)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, score_list, helpfulness_list, lengths = [], [], [], [], []
    for _helpfulness, _score, _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(vocab(_text), dtype=torch.int64)
        text_list.append(processed_text)
        score_list.append(_score)
        helpfulness_list.append(_helpfulness)
        lengths.append(processed_text.size(0))  # Store the length of each sequence

    # Pad the sequences
    padded_texts = pad_sequence(text_list, batch_first=True)

    # Get the max length after padding
    max_len = padded_texts.size(1)

    # Ensure that no sequence length exceeds the padded sequence length
    lengths = [min(length, max_len) for length in lengths]

    label_list = torch.tensor(label_list, dtype=torch.int64)
    score_list = torch.tensor(score_list, dtype=torch.float64)
    helpfulness_list = torch.tensor(helpfulness_list, dtype=torch.float64)

    return label_list.to(device), padded_texts.to(device), torch.tensor(lengths).to(device), score_list.to(device), helpfulness_list.to(device)

train_dataloader = DataLoader(
    train.to_numpy(), batch_size=128, shuffle=True, collate_fn=collate_batch
)

val_dataloader = DataLoader(
    val.to_numpy(), batch_size=128, shuffle=False, collate_fn=collate_batch
)