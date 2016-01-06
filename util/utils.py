#!/usr/bin/env python
# Martin Kersner, martin@company100.com
# 2016/01/06

from __future__ import print_function
import os
import sys

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
