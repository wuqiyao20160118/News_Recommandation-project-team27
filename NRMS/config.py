hyperParams = {
    'description': 'NRMS',
    'batch_size': 32,
    'num_workers': 4,
    'shuffle': False,
    'lr': 1e-5,
    'glove_path': './data/glove',
    'train_data_path': './data/train',
    'val_data_path': './data/train',
    'test_data_path': './data/train',
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
