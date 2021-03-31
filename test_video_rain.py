#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F


import networks
import utils
import matplotlib.pyplot as plt

def align_to_64(frame_i0, divide):
    [b, c, h, w] = frame_i0.shape

    h_pad = int(np.floor(h/divide)+1)*divide
    w_pad = int(np.floor(w/divide)+1)*divide

    frame_i0_pad = F.pad(frame_i0, pad = [0, w_pad-w, 0, h_pad-h], mode='replicate')

    return frame_i0_pad, h_pad-h, w_pad-w

def img_resize(frame_i0, sc, divide):
    h_in = frame_i0.shape[0]
    w_in = frame_i0.shape[1]
    h = int(math.floor(h_in * sc / divide)+1)*divide
    w = int(math.floor(w_in * sc / divide)+1)*divide
    frame_i0 = cv2.resize(frame_i0, (w, h))
    return frame_i0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    # parser.add_argument('-method', type=str, required=True, help='test model name')
    parser.add_argument('-model_name', type=str, required=True, help='test model name')
    parser.add_argument('-epoch', type=int, required=True, help='epoch')
    parser.add_argument('-streak_tag',          type=str,     default=1,               help='Whether the model handle rain streak')
    parser.add_argument('-haze_tag',            type=str,     default=1,               help='Whether the model handle haze')
    parser.add_argument('-flow_tag',            type=str,     default=1,               help='Whether the model handle haze')

    parser.add_argument('-dataset', type=str, default='Video_rain', help='dataset to test')
    parser.add_argument('-phase', type=str, default="train", choices=["train", "test"])
    parser.add_argument('-data_dir', type=str, default='/home/disk_wending/Data/Rain_Data', help='path to data folder')
    parser.add_argument('-list_dir', type=str, default='lists', help='path to list folder')

    parser.add_argument('-checkpoint_dir', type=str, default='./checkpoints/', help='path to checkpoint folder')
    # parser.add_argument('-task', type=str, required=True, help='evaluated task')
    parser.add_argument('-list_filename', type=str, default='./lists/video_rain_removal_test.txt', help='evaluated task')
    parser.add_argument('-redo', action="store_true", help='Re-generate results')
    parser.add_argument('-gpu', type=int, default=1, help='gpu device id')

    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 3  ## Inputs to TransformNet need to be divided by 4

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")



    model_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "model_epoch_%d.pth" % opts.epoch)
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)

    opts.rgb_max = 1.0
    opts.fp16 = False
    three_dim_model = networks.__dict__['TDARevModel'](opts, 3, 64, 16)
    

    # three_dim_model.load_state_dict(state_dict['three_dim_model'])

    this_state_dict = state_dict['three_dim_model']
    own_state = three_dim_model.state_dict()
    for name, param in this_state_dict.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        own_state[name].copy_(param)



    device = torch.device("cuda" if opts.cuda else "cpu")
    three_dim_model = three_dim_model.cuda()

    # three_dim_model.eval()
    three_dim_model.train()

    list_filename = './lists/video_rain_removal_test.txt'

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    for v in range(len(video_list)):
        video = video_list[v]

        print("Test %s on %s video %d/%d: %s" % (opts.dataset, opts.phase, v + 1, len(video_list), video))

        input_dir = os.path.join(opts.data_dir,  opts.phase, video)
        output_dir = os.path.join("./output", opts.model_name, "epoch_%d" % opts.epoch, video)
        output_input_dir = os.path.join("./output", opts.model_name, "input", video)

        print(input_dir)
        print(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not os.path.isdir(output_input_dir):
            os.makedirs(output_input_dir)    

        frame_list = glob.glob(os.path.join(input_dir, "*.png"))
        output_list = glob.glob(os.path.join(output_dir, "*.png"))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue

        for t in range(len(frame_list)-3, len(frame_list)-2):
            frame_i0 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t-3)))
            frame_i1 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t-2)))
            frame_i2 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t-1)))
            frame_i3 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t)))
            frame_i4 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t+1)))
            frame_i5 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t+2)))
            frame_i6 = utils.read_img(os.path.join(input_dir, frame_list[0][:-10] + "%06d.png" % (t+3)))


            sc = 0.9
          
            divide_size = 64
# 
            frame_i0 = img_resize(frame_i0, sc, divide_size)
            frame_i1 = img_resize(frame_i1, sc, divide_size)
            frame_i2 = img_resize(frame_i2, sc, divide_size)
            frame_i3 = img_resize(frame_i3, sc, divide_size)
            frame_i4 = img_resize(frame_i4, sc, divide_size)
            frame_i5 = img_resize(frame_i5, sc, divide_size)
            frame_i6 = img_resize(frame_i6, sc, divide_size)

            with torch.no_grad():
                frame_i0 = utils.img2tensor(frame_i0).cuda()
                frame_i1 = utils.img2tensor(frame_i1).cuda()
                frame_i2 = utils.img2tensor(frame_i2).cuda()
                frame_i3 = utils.img2tensor(frame_i3).cuda()
                frame_i4 = utils.img2tensor(frame_i4).cuda()
                frame_i5 = utils.img2tensor(frame_i5).cuda()
                frame_i6 = utils.img2tensor(frame_i6).cuda()
                

               
                [b, c, h, w] = frame_i0.shape


                frame_i0 = frame_i0.view(b, 1, c, h, w)
                frame_i1 = frame_i1.view(b, 1, c, h, w)
                frame_i2 = frame_i2.view(b, 1, c, h, w)
                frame_i3 = frame_i3.view(b, 1, c, h, w)
                frame_i4 = frame_i4.view(b, 1, c, h, w)
                frame_i5 = frame_i5.view(b, 1, c, h, w)
                frame_i6 = frame_i6.view(b, 1, c, h, w)

                frame_input = torch.cat((frame_i0, frame_i1, frame_i2, frame_i3, frame_i4, frame_i5, frame_i6), 1)

                frame_restr, frame_derain = three_dim_model.forward_both(frame_input*2-1)



            frame_pred = utils.tensor2img(torch.cat([frame_i3.view(1, c, h, w), frame_restr, frame_derain], dim = 3) )

            input_dir_idx = len(input_dir)
            output_filename = output_dir + frame_list[0][input_dir_idx:-10] + "%06d.png"%t
  
            # utils.save_img(frame_pred, output_filename)
            painter = (frame_restr[0].cpu().detach().numpy()).transpose(1,2,0)
            painter = np.clip((painter*0.5+0.5) * 255, 0, 255)
            painter_image = Image.fromarray(painter.astype(np.uint8))
            painter_image.save(output_filename)   

            frame_input = utils.tensor2img(frame_i3.view(1, c, h, w))
            output_input_filename = output_input_dir + frame_list[0][input_dir_idx:-10] + "%06d.png"%t
            utils.save_img(frame_input, output_input_filename)

    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
