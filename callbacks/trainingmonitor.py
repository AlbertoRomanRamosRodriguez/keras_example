from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath:str, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath # output plot file
        self.jsonPath = jsonPath # optional path to serialize the loss and accuracy
        self.startAt = startAt # starting epoch that training is 
                               # resumed at when using ctrl + c
                               # training
    
    def on_train_begin(self, logs={}):
        self.H = {} # initialization of the history dict

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read)

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    
    def on_epoch_end(self, epoch, logs={}):
        for (k,v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l
        
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.H))
            f.close()
        
        currentEpoch = len(self.H['loss'])

        if currentEpoch > 1:
            N = np.arange(0, currentEpoch)
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['accuracy'], label='train_acc')
            plt.plot(N, self.H['val_accuracy'], label='val_acc')
            plt.title(f"Training Loss and Accuracy [Epoch {currentEpoch}]")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            plt.savefig(self.figPath)
            plt.close()