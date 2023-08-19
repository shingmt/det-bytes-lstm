import os
import numpy as np
from tensorflow.python.keras.models import load_model, Model
from utils.utils import log
from main.data_helpers import load_data_x


class BytesModule:

    #? model config
    _model = None
    _model_path = ''
    _vocab_path = None
    _sequence_length = None


    def __init__(self, config):
        if config is None or 'model_path' not in config:
            log('[!][BytesModule] no `model_path` defined', 'warning')
            return
        self.change_config(config)
        log('[ ][BytesModule] model_path', self._model_path)

        self._vocab_path = config['vocab_path']
        self._sequence_length = config['sequence_length']

        """ Load model """
        if os.path.isfile(self._model_path):
            self._model = load_model(self._model_path)
        else: #? model_path not exist
            log('[!][BytesModule] `model_path` not exist', 'warning')
        # self._model.summary()

        return
    
    
    def change_config(self, config):
        if config is None:
            return

        #? if model_path is passed in config, load new model
        if 'model_path' in config and config['model_path'] != self._model_path:
            self._model_path = config['model_path']

            if not os.path.isfile(self._model_path): #? model_path not exist
                self._model = load_model(self._model_path)
            else: #? model_path not exist
                log('[!][BytesModule][change_config] `model_path` not exist', 'warning')
                self._model = None

        if 'sequence_length' in config and config['sequence_length'] != self._sequence_length:
            self._sequence_length = config['sequence_length']
        if 'vocab_path' in config and config['vocab_path'] != self._vocab_path:
            self._vocab_path = config['vocab_path']

        return


    def from_files(self, _map_ohash_inputs, callback):
        seq_datas = []
        for ohash,filepath in _map_ohash_inputs:
            content = open(filepath, 'r').read().strip()
            seq_datas.append(content)


        if self._model is None:
            log('[!][BytesModule][change_config] `model` not found', 'error')
            #? return empty result for each item
            result = {ohash: '' for ohash in _map_ohash_inputs.keys()}
            callback(result)
            return


        """ Infer """
        X = load_data_x(seq_datas, sequence_length=self._sequence_length,
                              vocabulary_inv_path=self._vocab_path)

        preds = [pred[0] for pred in self._model.predict(X)]
        lbl_preds = np.array([1 if pred > 0.5 else 0 for pred in preds]) #? only 1 or 0 (boolean result)

        print('[+][BytesModule][from_files] lbl_preds, preds', lbl_preds, preds)

        #? Callbacks on finish
        result = {}
        note = {}
        k = 0
        for ohash in _map_ohash_inputs.keys():
            result[ohash] = bool(int(lbl_preds[k]))
            note[ohash] = float(preds[k])
            k += 1

        #! Call __onFinishInfer__ when the analysis is done. This can be called from anywhere in your code. In case you need synchronous processing
        callback(result, note)

        # return lbl_preds, preds
        return