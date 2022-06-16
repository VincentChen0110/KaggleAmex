import copy
class early_stopper(object):
    def __init__(self, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        #self.val_preds = []
        #self.val_logits = []

    def earlystop(self, loss, value, model=None):#, preds, logits):
        """
        value: evaluation value on valiation dataset
        """
        cv = value
        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            #self.val_preds = preds
            #self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            #self.val_preds = preds
            #self.val_logits = logits
            self.count = 0