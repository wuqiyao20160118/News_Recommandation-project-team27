hyperParams = {
    'description': 'NRMS',
    'version': 'v1',
    'batch_size': 32,
    'num_workers': 4,
    'shuffle': False,
    'lr': 5e-4,
    'glove_path': './data/glove',
    'train_data_path': './data',
    'val_data_path': './data',
    'test_data_path': './data',
    'max_vocab_size': 50000,
    'model': {
        'head_num': 10,
        'embedding_size': 100,
        'hidden_size': 300,
        'q_size': 200,
    },
    'data': {
        'pos_num': 50,
        'neg_num': 4,
        'maxLen': 20
    }
}
