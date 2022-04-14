from subprocess import call
import pickle
import numpy as np
import os


import argparse
import pickle
import os
import numpy as np
from PIL import Image

import json
from collections import OrderedDict


class Config(object):
  def __init__(self, filename, update_config_string=""):
    lines = open(filename).readlines()
    # remove comments (starting with #)
    lines = [l if not l.strip().startswith("#") else "\n" for l in lines]
    lines = [l.split('#')[0] if '#' in l else l for l in lines]
    s = "".join(lines)
    print(s)
    self._entries = json.loads(s, object_pairs_hook=OrderedDict)
    if update_config_string != "":
      config_string_entries = json.loads(update_config_string, object_pairs_hook=OrderedDict)
      print("Updating given config with dict", config_string_entries)
      self._entries.update(config_string_entries)

  def has(self, key):
    return key in self._entries

  def _value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, dtype)
    if key in self._entries:
      val = self._entries[key]
      if isinstance(val, dtype):
        return val
      else:
        raise TypeError()
    else:
      assert default is not None
      return default

  def _list_value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, list)
      for x in default:
        assert isinstance(x, dtype)
    if key in self._entries:
      val = self._entries[key]
      assert isinstance(val, list)
      for x in val:
        assert isinstance(x, dtype)
      return val
    else:
      assert default is not None
      return default

  def bool(self, key, default=None):
    return self._value(key, bool, default)

  def str(self, key, default=None):
    if isinstance(default, str):
      default = str(default)
    return self._value(key, str, default)

  def int(self, key, default=None):
    return self._value(key, int, default)

  def float(self, key, default=None):
    return self._value(key, float, default)

  def dict(self, key, default=None):
    return self._value(key, dict, default)

  def int_key_dict(self, key, default=None):
    if default is not None:
      assert isinstance(default, dict)
      for k in list(default.keys()):
        assert isinstance(k, int)
    dict_str = self.str(key, "")
    if dict_str == "":
      assert default is not None
      res = default
    else:
      res = eval(dict_str)
    assert isinstance(res, dict)
    for k in list(res.keys()):
      assert isinstance(k, int)
    return res

  def int_list(self, key, default=None):
    return self._list_value(key, int, default)

  def float_list(self, key, default=None):
    return self._list_value(key, float, default)

  def str_list(self, key, default=None):
    return self._list_value(key, str, default)

  def dir(self, key, default=None):
    p = self.str(key, default)
    if p[-1] != "/":
      return p + "/"
    else:
      return p


def load_seqmap(seqmap_filename):
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      seq = "%04d" % int(fields[0])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
  return seqmap, max_frames

class CalibrationParameters:
    
    def __init__(self, filepath,img_size):
        calib_params = open(filepath, 'r').readlines()
        self.cam_projection_matrix_l_grey = self.parseLine(calib_params[1], (3, 4))
        self.cam_projection_matrix_l = self.parseLine(calib_params[2], (3, 4))
        self.cam_projection_matrix_r = self.parseLine(calib_params[3], (3, 4))

        self.fx = self.cam_projection_matrix_l[0, 0]
        self.fy = self.cam_projection_matrix_l[1, 1]
        self.cx = self.cam_projection_matrix_l[0, 2]
        self.cy = self.cam_projection_matrix_l[1, 2]
        self.bf = -self.cam_projection_matrix_l_grey[0,3]
        self.width = img_size[0]
        self.height = img_size[1]


        self.camera_matrix = np.asarray([self.fx, 0., self.cx, 0., self.fy, self.cy, 0., 0., 1.]).reshape((3, 3))

    def parseLine(self, line, shape):
        data = line.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data

def compute_poses_orb(sequence, calib_params, save_path, left_dir, right_dir, model_path, vocab_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    template_file = open(model_path + 'KITTI03.yaml', 'r')

    calibration_file = template_file.readlines()
    for i in range(len(calibration_file)):
        calibration_file[i] = calibration_file[i].replace('undefined_fx', str(calib_params.fx))
        calibration_file[i] = calibration_file[i].replace('undefined_fy', str(calib_params.fy))
        calibration_file[i] = calibration_file[i].replace('undefined_cx', str(calib_params.cx))
        calibration_file[i] = calibration_file[i].replace('undefined_cy', str(calib_params.cy))
        calibration_file[i] = calibration_file[i].replace('undefined_bf', str(calib_params.bf))
        calibration_file[i] = calibration_file[i].replace('undefined_width', str(calib_params.width))
        calibration_file[i] = calibration_file[i].replace('undefined_height', str(calib_params.height))

    calibration_path = save_path + sequence + '.yaml'

    open(calibration_path, 'w').writelines(calibration_file)

    call([model_path + 'stereo_kitti_old', vocab_path[:-1], calibration_path, left_dir, right_dir, save_path])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='path of config file to load', dest='config',
                        type=str, default='./config_default')
    args = parser.parse_args()
    config = Config(args.config)
    list_sequences, _ = load_seqmap(config.str('mots_seqmap_file'))


    for sequence in list_sequences:
            print(sequence)
            calibration_dir=config.dir('calibration_dir')

            left_dir = config.dir('left_img_dir') + sequence + '/'
            right_dir = config.dir('right_img_dir') + sequence + '/'

            img_size = Image.open(left_dir+'000000.png').size

            calibration_params = CalibrationParameters(calibration_dir + sequence + '.txt',img_size)


            print('computing poses..')
            compute_poses_orb(sequence, calibration_params, config.dir('orb_pose_savedir') + sequence + '/', left_dir, right_dir, config.dir('orbslam_modeldir'),
                                config.dir('orbslam_vocab_dir'))
            print('done.')



