from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd
import pickle

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.preprocess import preprocess

def yield_tokens(df):
    for _, sample in df.iterrows():
        yield sample.to_list()[2]


def build_loaders(train_path, val_path):
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

        return label_list.to(device), padded_texts.to(device), torch.tensor(lengths).to(device), score_list.to(
            device), helpfulness_list.to(device)

    train = preprocess(pd.read_csv(train_path))
    val = preprocess(pd.read_csv(val_path))

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    vocab = build_vocab_from_iterator(yield_tokens(train), specials=special_symbols)
    vocab.set_default_index(UNK_IDX)

    vocab_file = Path(os.getcwd()) / 'vocab.pickle'

    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

    sample = train['Text'][2]
    print(f"Sample: {sample}")
    print(f"Type: {type(sample)}")
    encoded = vocab(sample)
    print(encoded)

    torch.manual_seed(420)
    device = 'cpu'

    train_dataloader = DataLoader(
        train.to_numpy(), batch_size=128, shuffle=True, collate_fn=collate_batch
    )

    val_dataloader = DataLoader(
        val.to_numpy(), batch_size=128, shuffle=False, collate_fn=collate_batch
    )

    return train_dataloader, val_dataloader