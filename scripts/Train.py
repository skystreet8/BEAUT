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
from argparse import ArgumentParser
import os
logger = logging.getLogger('Train')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train(model, train_dataloader, optimizer, loss_fn, epoch):
    model.train()
    train_loss = 0
    logger.info(f'Epoch ::: {epoch}')
    for batch_idx, batch_data in enumerate(train_dataloader):
        reprs, labels = batch_data
        reprs, labels = reprs.to(DEVICE), labels.to(DEVICE)
        out = model(reprs)
        out = torch.sigmoid(out)
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
            repr, labels = batch_data
            repr, labels = repr.to(DEVICE), labels.to(DEVICE)
            out = model(repr)
            out = torch.sigmoid(out)
            loss = loss_fn(out, labels)
            val_loss += loss.item()
    return val_loss / (batch_idx + 1)


def predict(model, test_dataloader, threshold=0.5):
    model.eval()
    predictions = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            reprs, labels = batch_data
            labels = labels.numpy()
            logger.debug(labels.shape)
            test_labels.extend(list(labels))
            reprs = reprs.to(DEVICE)
            out = model(reprs)
            probs = torch.sigmoid(out)
            probs = probs.cpu().numpy()
            logger.debug(probs.shape)  # Should be (batch_size, 10117)
            batch_predictions = np.where(probs > threshold, 1, 0)
            predictions.extend(list(batch_predictions))
    return predictions, test_labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aug', action='store_true', default=False, help='Whether to train the Aug model.')
    args = parser.parse_args()
    LR = 0.0002
    BATCH_SIZE = 64
    NUM_FOLDS = 5
    NUM_EPOCHS = 100
    if not os.path.exists('../models'):
        os.mkdir('../models')

    dataset = SequenceDataset(fold=1, aug=args.aug)
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
        loss_fn = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        if args.aug:
            stopper = EarlyStopping(mode='lower', patience=5, filename=f'../models/ba_pred_DNN_aug_fold_{i + 1}.pth')
        else:
            stopper = EarlyStopping(mode='lower', patience=5, filename=f'../models/ba_pred_DNN_base_fold_{i + 1}.pth')
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, min_lr=5e-6)
        num_batches = len(train_dataloader)
        logger.info('-------Starting training-------')
        for ne in range(NUM_EPOCHS):
            train(model, train_dataloader, optimizer, loss_fn, ne + 1)
            val_loss = evaluate(model, val_dataloader, loss_fn)
            early_stop = stopper.step(val_loss, model)
            logger.info(f'Validation loss: {round(val_loss, 4)}')
            if early_stop:
                logger.info('Early stopped!')
                break
            scheduler.step(val_loss)
