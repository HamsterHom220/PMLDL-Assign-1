import mlflow
import mlflow.pytorch
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np

from tqdm.autonotebook import tqdm
import torch
from torch.optim import SGD
from torch import nn
from pathlib import Path
import pickle

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from biGRU import TextClassificationModel
from datasets.dataload_utils import build_loaders


def train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    for i, batch in loop:
        labels, texts, offsets, scores, helpfulness = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(texts, offsets)
        # loss calculation
        loss = loss_fn(outputs, labels.long())

        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss / (i * len(labels))})

    with mlflow.start_run(nested=True) as run:
        # Log training loss to MLflow
        mlflow.log_metric("train_loss", train_loss / len(loader))


def val_one_epoch(
        model,
        loader,
        loss_fn,
        epoch_num=-1,
        best_so_far=0.0,
        ckpt_path='best.pt'
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            labels, texts, offsets, scores, helpfulness = batch

            # forward pass
            outputs = model(texts, offsets)

            # loss calculation
            loss = loss_fn(outputs, labels.long()).item()

            # prediction and accuracy calculation
            predicted = outputs.argmax(dim=1, keepdim=True)
            c = predicted.eq(labels.view_as(predicted)).sum().item()
            correct += c

            # increment total by the actual batch size
            t = len(labels)
            total += t

            val_loss += loss
            loop.set_postfix({"loss": val_loss / total, "acc": c / t})

        if correct / total > best_so_far:
            torch.save(model.state_dict(), ckpt_path)
            return correct / total

    with mlflow.start_run(nested=True) as run:
        # Log validation loss and accuracy to MLflow
        mlflow.log_metric("val_loss", val_loss / len(loader))
        mlflow.log_metric("val_accuracy", correct / total)

    return best_so_far

if __name__ == "__main__":
    with mlflow.start_run() as run:
        device = 'cpu'

        data_dir = Path('~/MLOps/Assign1/data/processed')
        train_dataloader, val_dataloader = build_loaders(data_dir / 'train.csv', data_dir / 'val.csv')

        epochs = 1
        lr = 0.01
        model = TextClassificationModel(6, 40000).to(device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)

        best = -float('inf')
        for epoch in range(epochs):
            train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
            best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best)

        # Define input/output schema for logging
        input_schema = Schema(
            [
                TensorSpec(np.dtype(np.int64), (128, 538), name='texts'),  # Input tensor shape: (batch_size, seq_length)
                TensorSpec(np.dtype(np.int64), (128,), name='offsets')  # Target labels
            ]
        )
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 6))])  # Output shape: (batch_size, num_classes)
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log the best model to MLflow
        ckpt = torch.load("best.pt")
        model.load_state_dict(ckpt)
        model_file = '/home/hamsterhom220/MLOps/Assign1/models/trained_model.pickle'
        print(os.path.exists(model_file))
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            signature=signature,
            registered_model_name="1",
            artifact_path="model-1"
        )