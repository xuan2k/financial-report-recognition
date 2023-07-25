"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lgpma_pub.py
# Abstract       :    Model settings for LGPMA detector on PubTabNet

# Current Version:    1.0.1
# Date           :    2022-09-05
##################################################################################################
"""

_base_ = "./lgpma_base.py"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file='/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/data/datalist_train_detection_pubtabnet.json',
        img_prefix='path/to/PubTabNet'),
    # According to the evaluation metric, select the appropriate validation dataset format.
    val=dict(
        ann_file='path/to/validation.json',
        img_prefix='path/to/PubTabNet'),
    test=dict(
        samples_per_gpu=1,
        ann_file='/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/data/PubTabNet_2.0.0_val.jsonl',
        img_prefix='/home/xuan/Project/OCR/result/demo/rgb/0002.png')
)

# yapf:enable
# runtime settings

checkpoint_config = dict(interval=1, filename_tmpl='/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/pretrained/maskrcnn-lgpma-pub-e12-pub.pth')
# checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/maskrcnn-lgpma-pub-e{}.pth')

work_dir = 'path/to/workdir'


# Online evaluation or batch evaluation
evaluation_metric = "TEDS"   # change to "hmean" for aligned bboxes evaluation

evaluation = dict(
    eval_func_params=dict(
        ENLARGE_ANN_BBOXES=True,
        IOU_CONSTRAINT=0.5
    ),
    metric=evaluation_metric,
    by_epoch=True,
    interval=1,
    eval_mode="general",
    save_best=evaluation_metric,
    rule='greater',
)
