#!/usr/bin/env python
# Martin Kersner, martin@company100.com
# 2016/01/05

import os
import sys
import time
import caffe
import scipy.io

from skimage.io import imread, imsave
from skimage.color import label2rgb
from skimage import img_as_ubyte
#from PIL import Image

from util.init_VOC2012_TEST import *
from util.preprocess_image import *
from util.utils import *

def main():
  config = {}
  config['imageset'] = 'test'
  config['cmap']= './voc_gt_cmap.mat'
  config['gpuNum'] = 0
  config['Path.CNN.caffe_root'] = './caffe'
  config['save_root'] = './results'

  # cache FCN-8s results
  config['write_file'] = 1 # used to be 1
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
  log('start caching FCN-8s results and score')
  
  ## initialization
  cmap = scipy.io.loadmat(config['cmap'])['cmap']

  ## initialize caffe
  log('initializing caffe..')
  caffe.set_mode_gpu()
  caffe.set_device(config['gpuNum'])
  net = caffe.Net(config['Path.CNN.model_proto'], config['Path.CNN.model_data'], caffe.TEST)
  log('done')
  
  ## initialize paths
  save_res_dir = os.path.join(config['save_root'], 'FCN8s/results')
  save_res_path = os.path.join(save_res_dir, '%s.png')
  save_score_dir = os.path.join(config['save_root'], 'FCN8s/scores')
  save_score_path = os.path.join(save_score_dir, '%s.npy')
  
  ## create directory
  if config['write_file']:
    create_dir(save_res_dir)
    create_dir(save_score_dir)
  
  ## load image set
  ids = textread(VOCopts['seg.imgsetpath'] % config['imageset'])
  
  for i in range(2):
  #for i in range(len(ids)):
      log('progress: {}/{} [{}]...'.format(i, len(ids), ids[i]))
      start = time.clock()
  
      # read image
      I = img_as_ubyte(imread(VOCopts['imgpath'] % ids[i])) # TODO does load correctly?
      #I = Image.open(VOCopts['imgpath'] % ids[i])
      input_data = preprocess_image(I, config['im_sz']) 

      net.blobs['data'].reshape(1, *input_data.shape)
      net.blobs['data'].data[...] = input_data 
      net.forward()
      result = net.blobs['upscore'].data[0]

      tr_result = result.transpose((1,2,0))
      score = tr_result[0:I.shape[0], 0:I.shape[1], :]
      result_seg = np.argmax(score, axis=2)
      result_seg -= 1 # TODO necessary?
      
      if config['write_file']:
        imsave(save_res_path % ids[i], label2rgb(result_seg, colors=cmap))
        np.save(save_score_path % ids[i], score)

      end = time.clock()
      print str(end - start) + " s"

if __name__ == '__main__':
  main()
