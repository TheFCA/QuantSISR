# Fernando Carrió: Based on code: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
# 
#
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=30, min_delta=0.001, mode='rel'):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.best_loss = val_loss
        else:
            if self.mode == 'rel':
                if self.best_loss*(1-self.min_delta)> val_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1            
                    print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
                    if self.counter >= self.patience:
                        print('INFO: Early stopping')
                        self.early_stop = True
            else: # Self mode is abs
                if self.best_loss - val_loss > self.min_delta:
                    self.best_loss = val_loss
                    self.counter = 0
                elif self.best_loss - val_loss < self.min_delta:
                    self.counter += 1
                    print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
                    if self.counter >= self.patience:
                        print('INFO: Early stopping')
                        self.early_stop = True