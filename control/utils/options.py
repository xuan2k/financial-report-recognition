# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class FOCROptions:
    def __init__(self):
     self.parser = argparse.ArgumentParser(description="FOCR options")

     # PATHS     
     self.parser.add_argument("--img_dir", 
                              type=str,
                              help="path to the image directory or a single file",
                              default="/home/xuan/Project/OCR/result/demo/rgb/0002.png")
     
     self.parser.add_argument("--save_dir", 
                              type=str,
                              help="path to save the result data",
                              default="/home/xuan/Project/OCR/code/git_code/DAVAR-Lab-OCR/demo/table_recognition/lgpma/result")

     # DATA options
     self.parser.add_argument("--model_name",
                              type=str,
                              help="the name of the folder to save the model in",
                              default="mdp")
     
     self.parser.add_argument("--png",
                              help="if set, set image extension as png files (instead of jpgs)",
                              action="store_true")
     
     self.parser.add_argument("--height",
                              type=int,
                              help="input image height",
                              default=192)
     
     self.parser.add_argument("--width",
                              type=int,
                              help="input image width",
                              default=640)
     
     self.parser.add_argument("--visualize", 
                              help="visualize the output on the image",
                              action="store_true")


     # TRAINING OPTIMIZATION options
     self.parser.add_argument("--batch_size",
                              type=int,
                              help="batch size",
                              default=12)
     
     self.parser.add_argument("--learning_rate",
                              type=float,
                              help="learning rate",
                              default=1e-4)
     
     self.parser.add_argument("--num_epochs",
                              type=int,
                              help="number of epochs",
                              default=20)
     
     self.parser.add_argument("--scheduler_step_size",
                              type=int,
                              help="step size of the scheduler",
                              default=15)
     
     self.parser.add_argument("--starting_epochs",
                              type=int,
                              help="starting epoch number",
                              default=0)
          
     # MODEL options
     self.parser.add_argument("--checkpoint",
                              type=str,
                              help="path to checkpoint file",
                              default="./pretrained/checkpoint.pth")
     
     self.parser.add_argument("--config",
                              type=str,
                              help="path to model config",
                              default="./config/model.yml")
     
     self.parser.add_argument("--device",
                              type=str,
                              help="Choose device to run model",
                              default="cuda:0")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
