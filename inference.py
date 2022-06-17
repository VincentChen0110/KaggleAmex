import torch
import numpy as np
import pandas as pd
import gc
from metric import amex_metric
from model.model import gru_model
from nntools.earlystopper import early_stopper

PATH_TO_DATA = '/home/vincent0110/test/KaggleAmex/data/amex-data-for-transformers-and-rnns/data/'
PATH_TO_MODEL = '/home/vincent0110/test/KaggleAmex/model/'

start = 0; end = 0
sub = pd.read_csv('/home/vincent0110/test/KaggleAmex/data/sample_submission.csv')
NUM_FILES = 20
# REARANGE SUB ROWS TO MATCH PROCESSED TEST FILES
sub['hash'] = sub['customer_ID'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
test_hash_index = np.load(f'{PATH_TO_DATA}test_hashes_data.npy')
sub = sub.set_index('hash').loc[test_hash_index].reset_index(drop=True)

params = {
    'model': 'gru_model',
    'batch_size': 512,
    'lr': 0.002,
    'wd': 1e-5,
    'device': 'cuda:6',
    'early_stopping': 4,
    'n_fold': 5,
    'seed': 42,
    'max_epochs': 20,
}
device = params['device']
for k in range(NUM_FILES):
    # LOAD TEST DATA
    print(f'Inferring Test_File_{k+1}')
    X_test = np.load(f'{PATH_TO_DATA}test_data_{k+1}.npy')
    end = start + X_test.shape[0]

    # BUILD MODEL
    model = eval(params['model'])(X_test.shape[-1]).to(device)
    
    # INFER 5 FOLD MODELS
    model.load_state_dict(torch.load(f'{PATH_TO_MODEL}gru_fold_1.h5'))
    test_predictions = torch.zeros(X_test.shape[0], 2).to(device).float()
    test_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(np.ones(X_test.shape[0]),
                                                                            num_samples=X_test.shape[0], replacement=False)
    test_dataloader = torch.utils.data.DataLoader(np.array(range(X_test.shape[0])), batch_size=params['batch_size'], num_workers=0,
                                                    sampler=test_sample_strategy, drop_last=False)
    
    for j in range(1,5):
        model.load_state_dict(torch.load(f'{PATH_TO_MODEL}gru_fold_{j+1}.h5'))
        with torch.no_grad():
            for step, input_seeds in enumerate(test_dataloader):
                batch_inputs = torch.from_numpy(X_test[input_seeds]).to(device).float()
                model.hidden_state = model.init_hidden(len(input_seeds), device)
                test_batch_logits = model(batch_inputs)
                test_predictions[input_seeds] = test_predictions[input_seeds] + torch.softmax(test_batch_logits, dim=-1)
                #test_batch_pred = torch.sum(torch.argmax(test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 50 == 0:
                    print('In fold {} test batch:{:04d}'.format(j+1, step))
    test_predictions /= 5.0

    # SAVE TEST PREDICTIONS
    sub.loc[start:end-1,'prediction'] = test_predictions[:, 1].cpu().numpy()
    start = end
    
    # CLEAN MEMORY
    del model, X_test
    gc.collect()

sub.to_csv('submission.csv',index=False)