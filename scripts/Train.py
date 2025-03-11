import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import DNNPredictor
from early_stopping import EarlyStopping
from dataset import SequenceDataset
import os
from sklearn.metrics import f1_score
logger = logging.getLogger('Train')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train(model, train_dataloader, optimizer, loss_fn, epoch):
    model.train()
    train_loss = 0
    logger.info(f'Epoch ::: {epoch}')
    for batch_idx, batch_data in enumerate(train_dataloader):
        reprs, labels, _ = batch_data
        reprs, labels = reprs.to(DEVICE), labels.to(DEVICE)
        out = model(reprs)
        labels = labels.squeeze(dim=1)
        loss = loss_fn(out, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f'Training loss: {round(train_loss / len(train_dataloader), 4)}')


def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            repr, labels, _ = batch_data
            repr, labels = repr.to(DEVICE), labels.to(DEVICE)
            out = model(repr)
            labels = labels.squeeze(dim=1)
            loss = loss_fn(out, labels)
            val_loss += loss.item()
    return val_loss / (batch_idx + 1)


def predict(model, test_dataloader, threshold=0.5):
    model.eval()
    predictions = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs, labels, _ = batch_data
            labels = labels.squeeze(dim=1)
            labels = labels.numpy()
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.softmax(out, dim=1)
            probs = probs.cpu().numpy()
            logger.debug(probs.shape)
            batch_predictions = np.argmax(probs, axis=1)
            predictions.extend(list(batch_predictions))
    return predictions, test_labels


def eval_metric(model, val_dataloader):
    model.eval()
    predictions = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            reprs, labels, _ = batch_data
            labels = labels.numpy()
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.softmax(out, dim=1)
            probs = probs.cpu().numpy()
            logger.debug(probs.shape)  # Should be (batch_size, 10117)
            batch_predictions = np.argmax(probs, axis=1)
            predictions.extend(list(batch_predictions))
    return predictions, test_labels


if __name__ == '__main__':
    LR = 0.0002
    BATCH_SIZE = 64
    NUM_FOLDS = 5
    NUM_EPOCHS = 100
    if not os.path.exists('../models'):
        os.mkdir('../models')

    dataset = SequenceDataset(fold=1)
    test_dataset = Subset(dataset, indices=dataset.test_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    for i in range(NUM_FOLDS):
        logger.info(f'-------Fold {i + 1}-------')
        dataset.set_fold(i + 1)
        train_dataset = Subset(dataset, indices=dataset.train_ids)
        val_dataset = Subset(dataset, indices=dataset.val_ids)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        model = DNNPredictor(1280, [256, 32])
        model.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        stopper = EarlyStopping(mode='higher', patience=5, filename=f'../models/BEAUT_aug_fold_{i + 1}.pth')
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, min_lr=5e-6)
        num_batches = len(train_dataloader)
        logger.info('-------Starting training-------')
        for ne in range(NUM_EPOCHS):
            train(model, train_dataloader, optimizer, loss_fn, ne + 1)
            # val_loss = evaluate(model, val_dataloader, loss_fn)
            val_preds, val_labels = eval_metric(model, val_dataloader)
            f1 = f1_score(val_labels, val_preds)
            early_stop = stopper.step(f1, model)
            # logger.info(f'Validation loss: {round(val_loss, 4)}')
            logger.info(f'Validation F1-score: {round(f1, 4)}')
            if early_stop:
                logger.info('Early stopped!')
                break
            scheduler.step(f1)
