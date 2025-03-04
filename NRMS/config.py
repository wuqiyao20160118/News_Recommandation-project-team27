hyperParams = {
    'description': 'NRMS',
    'version': 'category',
    'batch_size': 32,
    'num_workers': 1,
    'shuffle': False,
    'lr': 5e-4,
    'glove_path': './data/glove',
    'train_data_path': './data/train',
    'val_data_path': './data/val',
    'test_data_path': './data/val',
    'max_vocab_size': 50000,
    'model': {
        'head_num': 10,
        'embedding_size': 300,
        'hidden_size': 300,
        'q_size': 200,
    },
    'data': {
        'pos_num': 50,
        'neg_num': 4,
        'maxLen': 30,
        'wordLen': 60,
        'titleLen': 30
    },
    'deep_cross': {
        'cross_num': 2,
        'cross_parameterization': 'vector',
        'l2_reg_linear': 0.00001,
        'l2_reg_cross': 0.00001,
        'l2_reg_dnn': 0,
        'init_std': 0.0001,
        'dnn_dropout': 0,
        'dnn_activation': 'relu',
        'dnn_use_bn': False,
        'task': 'binary'
    },
    'checkpoint_path_category': './checkpoints/NRMS/category/epoch=6-auroc=0.71.ckpt',
    'checkpoint_path': './checkpoints/NRMS/v5/epoch=8-auroc=0.71.ckpt',
    'checkpoint_path_title': './checkpoints/NRMS/v1/epoch=3-auroc=0.71.ckpt',
}
