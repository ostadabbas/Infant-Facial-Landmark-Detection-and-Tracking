# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    filenames, nme, predictions = function.inference(config, test_loader, model)
    
    # Save coordinate predictions externally in an ad hoc manner.
    output_dir = os.path.join(final_output_dir, 'predictions')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 
    
    pred_array = predictions.numpy()
    
    for i in range(len(filenames)):
        with open(os.path.join(output_dir, os.path.splitext(filenames[i])[0] + '.txt'), 'w') as output_text:
            output_text.write('file: ' + filenames[i] + '\n')
            output_text.write('x: ' + str(list(pred_array[i, :, 0])) + '\n')
            output_text.write('y: ' + str(list(pred_array[i, :, 1])) + '\n') 
            output_text.write('min_box: ' + str([np.min(pred_array[i, :, 0]),
                                                 np.min(pred_array[i, :, 1]),
                                                 np.max(pred_array[i, :, 0]),
                                                 np.max(pred_array[i, :, 1])]))
                                                 
    print('Predictions saved to', output_dir)

    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

