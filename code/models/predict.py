import torch
import pickle
from pathlib import Path
from pandas import DataFrame

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.preprocess import preprocess

def predict(entry):
    # Function to convert words to indices while skipping unknown words
    def text_to_indices(text: list[str], vocab):
        indices = []
        for word in text:
            if word in vocab:
                indices.append(vocab[word])  # Add the word index if it exists in vocab
            else:
                print(f"Word '{word}' not in vocabulary, skipping.")
        return indices

    # model_file = Path(os.getcwd()).parent.parent.parent/'models/trained_model.pickle'
    model_file = '/api/models/trained_model.pickle'
    device = 'cpu'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        processed_entry = preprocess(entry)
        text = processed_entry['Text'][0]
        # score = processed_entry['Score']
        # helpfulness = processed_entry['Helpfulness']

        #vocab_file = Path(os.getcwd()).parent.parent/'models/vocab.pickle'
        vocab_file = '/api/code/models/vocab.pickle'
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)

            # Process the text to convert it into a tensor of token indices
            token_indices = text_to_indices(text, vocab)
            # Ensure processed_text is not empty after skipping unknown words
            if len(token_indices) == 0:
                print("No valid tokens found after checking vocabulary.")
                return None

            processed_text = torch.tensor(token_indices, dtype=torch.int64).to(device)

            # if processed_text.size(0) == 0:  # Handle empty text case
            #     processed_text = torch.tensor([vocab('<pad>')], dtype=torch.int64).to(device)

            offsets = torch.tensor([0], dtype=torch.int64).to(device)  # Single entry, so offset is 0
            processed_text = processed_text.unsqueeze(0)

            with torch.no_grad():  # Disable gradient calculation
                # Forward pass
                output = model(processed_text, offsets)

                # Get the predicted class (index of the max log-probability)
                predicted_class = torch.argmax(output, dim=1).item()

                # define categories indices
                cat2idx = {
                    'toys games': 0,
                    'health personal care': 1,
                    'beauty': 2,
                    'baby products': 3,
                    'pet supplies': 4,
                    'grocery gourmet food': 5,
                }
                # define reverse mapping
                idx2cat = {
                    v: k for k, v in cat2idx.items()
                }

                return idx2cat[predicted_class]

# (Title,Helpfulness,Score,Text) -> Category
# sample_title = "Something"
# sample_text = "You can use other tools than suggested. You can also add another tools to the pipeline, but  the specified stages should be present in the pipeline.Any dataset and models could be used"
# print(predict(DataFrame({'Title': [sample_title], 'Helpfulness': ['0/0'], 'Score': [0], 'Text': [sample_text]})))