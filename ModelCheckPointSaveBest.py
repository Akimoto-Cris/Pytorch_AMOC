#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.28 : 下午 10:11
@File Name: ModelCheckPointSaveBest.py
@Software: PyCharm
-----------------
"""
from ignite.handlers import ModelCheckpoint
import os
from ignite.engine import Events
from typing import Callable


class ModelCheckpointSaveBest(ModelCheckpoint):
    """"
        Extends class`ignite.handlers.ModelCheckpoint with option to provide a custom save method,
        saving the final model after training ends and saving a model if an exception is raised during training

        Reference: [](https://github.com/pytorch/ignite/issues/387#issuecomment-479666397)
        """

    def __init__(self, *args, save_method: Callable[[], None] = None, save_on_exception: str = True,
                 save_on_completed: str = True, **kwargs):
        self._save_method = save_method
        self._save_on_completed = save_on_completed
        self._save_on_exception = save_on_exception

        super(ModelCheckpointSaveBest, self).__init__(*args, **kwargs)

    def _internal_save(self, obj, path):
        if self._save_method is not None:
            self._save_method(obj, path)
        else:
            super(ModelCheckpointSaveBest, self)._internal_save(obj, path)

    def _on_exception(self, engine, exception, to_save):
        for name, obj in to_save.items():
            fname = '{}_{}_{}{}.pth'.format(self._fname_prefix, name, self._iteration, "_on_exception")
            path = os.path.join(self._dirname, fname)
            if os.path.exists(path):
                os.remove(path)
            self._save(obj=obj, path=path)

    def _on_completed(self, engine, to_save):
        for name, obj in to_save.items():
            fname = '{}_{}_{}{}.pth'.format(self._fname_prefix, name, self._iteration, "_on_completed")
            path = os.path.join(self._dirname, fname)
            if os.path.exists(path):
                os.remove(path)
            self._save(obj=obj, path=path)

    def attach(self, engine, model_dict):
        """
        Attaches the model saver to an engine object

        Args:
            engine (Engine): engine object
            model_dict (dict): A dict mapping names to objects, e.g. {'mymodel': model}
        """
        engine.add_event_handler(Events.EPOCH_COMPLETED, self, model_dict)
        if self._save_on_completed:
            engine.add_event_handler(Events.COMPLETED, self._on_completed, model_dict)
        if self._save_on_exception:
            engine.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception, model_dict)