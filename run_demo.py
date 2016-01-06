#!/usr/bin/python
# Martin Kersner, martin@company100.com
# 2016/01/05

from __future__ import print_function
import os
import sys
import caffe
import scipy.io
from skimage.io import imread
from util.init_VOC2012_TEST import *

def main():
  config = {}
  config['imageset'] = 'test'
  config['cmap']= './voc_gt_cmap.mat'
  config['gpuNum'] = 0
  config['Path.CNN.caffe_root'] = './caffe'
  config['save_root'] = './results'

  # cache FCN-8s results
  config['write_file'] = 0 # used to be 1
  config['Path.CNN.script_path'] = './FCN'
  config['Path.CNN.model_data'] = os.path.join(config['Path.CNN.script_path'], 
                                  'fcn-8s-pascal.caffemodel')
  config['Path.CNN.model_proto'] = os.path.join(config['Path.CNN.script_path'], 
                                   'fcn-8s-pascal-deploy.prototxt')
  config['im_sz'] = 500
  
  cache_FCN8s_results(config)
  
  # generate EDeconvNet+CRF results
  config['write_file'] = 1 
  config['edgebox_cache_dir'] = './data/edgebox_cached/VOC2012_TEST'
  config['Path.CNN.script_path'] = './DeconvNet'
  config['Path.CNN.model_data'] = os.path.join(config['Path.CNN.script_path'],
                                  'DeconvNet_trainval_inference.caffemodel')
  config['Path.CNN.model_proto'] = os.path.join(config['Path.CNN.script_path'], 
                                   'DeconvNet_inference_deploy.prototxt')
  config['max_proposal_num'] = 50
  config['im_sz'] = 224
  config['fcn_score_dir'] = './results/FCN8s'
  
  #generate_EDeconvNet_CRF_results(config)

def cache_FCN8s_results(config):
  log('start caching FCN-8s restuls and score')
  
  ## initialization
  #load(config.cmap)
  cmap = scipy.io.loadmat(config['cmap'])
  #init_VOC2012_TEST

  ## initialize caffe
  #addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'))
  log('initializing caffe..')
  #caffe('init', config['Path.CNN.model_proto'], config['Path.CNN.model_data'])
  #caffe.set_mode_gpu()
  #caffe.set_device(config['gpuNum'])
  #net = caffe.Net(prototxt, model, caffe.TEST)
  log('done')
  
  ## initialize paths
  save_res_dir = os.path.join(config['save_root'], 'FCN8s/results')
  save_res_path = os.path.join(save_res_dir, '%s.png')
  save_score_dir = os.path.join(config['save_root'], 'FCN8s/scores')
  save_score_path = os.path.join(save_score_dir, '%s.mat')
  
  ## create directory
  if config['write_file']:
    create_dir(save_res_dir)
    create_dir(save_score_dir)
  
  ## load image set
  path_to_dataset = '/home/martin/datasets/VOC2011/Test/VOCdevkit/VOC2011'
  ids = textread(VOCopts['seg.imgsetpath'] % config['imageset'])
  #print(VOCopts['seg.imgsetpath'] % config['imageset'])
  
  #for i=1:length(ids)
  for i in range(len(ids)):
      log('progress: {}/{} [{}]...'.format(i, len(ids), ids[i]))
      #tic
  
      # read image
      I = imread(VOCopts['imgpath'] % ids[i])
  
      input_data = preprocess_image(I, config.im_sz) 
      #cnn_output = caffe('forward', input_data)
      #
      #result = cnn_output{1}
      #
      #tr_result = permute(result,[2,1,3])
      #score = tr_result(1:size(I,1),1:size(I,2),:)
      #
      #[~, result_seg] = max(score,[], 3)   
      #result_seg = uint8(result_seg-1)
      #
      #if config.write_file
      #    imwrite(result_seg, cmap, sprintf(save_res_path, ids{i}))
      #    save(sprintf(save_score_path, ids{i}), 'score')
      #else
      #    subplot(1,2,1)
      #    imshow(I)
      #    subplot(1,2,2)
      #    result_seg_im = reshape(cmap(int32(result_seg)+1,:),[size(result_seg,1),size(result_seg,2),3])
      #    imshow(result_seg_im)
      #    waitforbuttonpress        
      #end
      #fprintf(' done [%f]\n', toc)

def create_dir(dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

def log(*msg):
  print("LOG: ", *msg, file=sys.stderr)

def textread(file_name):
  ids = []
  with open(file_name, 'r') as f:
    for line in f:
     ids.append(line.strip()) 

  return ids

if __name__ == '__main__':
  main()
