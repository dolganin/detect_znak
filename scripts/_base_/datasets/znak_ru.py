znak_textrecog_data_root = '/workspace/datasets/Rus'

znak_textrecog_train = dict(
    type='OCRDataset',
    data_root=znak_textrecog_data_root,
    ann_file='train/train_rec.json',
    pipeline=None)

znak_textrecog_test = dict(
    type='OCRDataset',
    data_root=znak_textrecog_data_root,
    ann_file='val/test_rec.json',
    test_mode=True,
    pipeline=None)