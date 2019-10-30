#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.27 : 下午 7:29
@File Name: train.py
@Software: PyCharm
-----------------
"""
import torch
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping
from test import compute_cmc
from torchnet.logger import VisdomLogger
from ModelCheckPointSaveBest import ModelCheckpointSaveBest
from typing import Iterable


def train_sequence(model: torch.nn.Module,
                   contrast_criterion: torch.nn.Module,
                   class_criterion_A: torch.nn.Module,
                   class_criterion_B: torch.nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   opt, printInds: Iterable = None) -> (torch.nn.Module, dict, dict):
    optimizer = optim.Adam(model.parameters(), lr=opt.learningRate)
    confusion_logger = VisdomLogger('heatmap', port=8097, opts={
        'title': 'simMat',
        'columnnames': list(range(len(train_loader.dataset))),
        'rownames': list(range(len(train_loader.dataset)))
    })
    if printInds is None:
        printInds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    def iterate_func(engine, batch):
        model.train()
        optimizer.zero_grad()
        inputA, inputB, target, personA, personB, _, _, _ = batch
        if len(inputA.shape) == len(inputB.shape) == 4:
            inputA = torch.unsqueeze(inputA, 0)
            inputB = torch.unsqueeze(inputB, 0)
        target = torch.unsqueeze(target, 0)
        personA = torch.squeeze(personA, 2)
        personB = torch.squeeze(personB, 2)
        assert inputA.shape[1] == inputB.shape[1] == opt.sampleSeqLength, \
            ValueError(f"inputA {inputA.shape}, inputB {inputB.shape}, required seq lenth {opt.sampleSeqLength}")
        if torch.cuda.is_available():
            inputA = inputA.cuda().float()
            inputB = inputB.cuda().float()
            target = target.cuda().float()
            personA = personA.cuda().float()
            personB = personB.cuda().float()
        distance, outputA, outputB = model(inputA, inputB)
        contrast_loss = contrast_criterion(distance, target)

        class_loss_A = class_criterion_A(outputA, personA)
        class_loss_B = class_criterion_B(outputB, personB)
        loss = contrast_loss + class_loss_A + class_loss_B
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.item(), contrast_loss.item(), class_loss_A.item(), class_loss_B.item()

    trainer = Engine(iterate_func)
    train_history = {'cnst': [], 'ceA': [], 'ceB': [], 'ttl': []}
    val_history = {'avgSame': [], 'avgDiff': [], 'cmc': [], 'simMat': []}
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'ttl')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'cnst')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'ceA')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'ceB')
    score_func = lambda engine: - engine.state.metrics['ttl']
    checkpoint_handler = ModelCheckpointSaveBest(opt.checkpoint_path,
                                                 filename_prefix=opt.saveFileName,
                                                 score_function=score_func,
                                                 require_empty=False)
    stop_handler = EarlyStopping(patience=15, trainer=trainer,
                                 score_function=score_func)

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        engine.state.iteration = opt.resume_epoch * len(engine.state.dataloader)
        engine.state.epoch = opt.resume_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_log(engine: Engine):
        avg_ttl = engine.state.metrics['ttl']
        avg_cnst = engine.state.metrics['cnst']
        avg_ceA = engine.state.metrics['ceA']
        avg_ceB = engine.state.metrics['ceB']
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch[{engine.state.epoch}] lr={lr:.2e}:\tTTL={avg_ttl:.3f}\tContrast={avg_cnst:.3f}\t"
              f"CrossEntA={avg_ceA:.3f}\tCrossEntB={avg_ceB:.3f}")

    @trainer.on(Events.ITERATION_COMPLETED)
    def adjust_lr(engine):
        # learning rate decay
        if engine.state.iteration % opt.lr_decay == 0 and engine.state.iteration >= 20000:
            lr = opt.learningRate * (0.1 ** (engine.state.iteration // opt.lr_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def on_complete(engine, dataloader, mode, history_dict):
        if not engine.state.epoch % opt.samplingEpochs:
            cmc, simMat, _, avgSame, avgDiff = compute_cmc(
                dataloader.dataset,
                dataloader.dataset.inds_set,
                model,
                opt.sampleSeqLength)

            metrics = {
                "cmc": cmc,
                "simMat": simMat,
                "avgSame": avgSame,
                "avgDiff": avgDiff
            }

            outString = ' '.join((str(torch.floor(cmc[c])) for c in printInds))

            print(f"{mode} Result: Epoch[{engine.state.epoch}]- Avg Same={avgSame:.3f}\tAvg Diff={avgDiff:.3f}")
            print(outString)

            confusion_logger.log(simMat)
            if mode == "Validation":
                for key in val_history.keys():
                    history_dict[key].append(metrics[key])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, on_complete, train_loader, 'Training', train_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, on_complete, test_loader, 'Validation', val_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, stop_handler)
    checkpoint_handler.attach(trainer, model_dict={"amoc": model})

    trainer.run(train_loader, max_epochs=opt.nEpochs)

    return model, trainer_log, val_history
