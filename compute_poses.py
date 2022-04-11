from subprocess import call
import pickle
import numpy as np
import os


import numpy as np

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
    
    def __init__(self, filepath):
        calib_params = open(filepath, 'r').readlines()
        self.cam_projection_matrix_l_grey = self.parseLine(calib_params[1], (3, 4))
        self.cam_projection_matrix_l = self.parseLine(calib_params[2], (3, 4))
        self.cam_projection_matrix_r = self.parseLine(calib_params[3], (3, 4))

        self.fx = self.cam_projection_matrix_l[0, 0]
        self.fy = self.cam_projection_matrix_l[1, 1]
        self.cx = self.cam_projection_matrix_l[0, 2]
        self.cy = self.cam_projection_matrix_l[1, 2]
        self.bf = -self.cam_projection_matrix_l_grey[0,3]

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
        calibration_file[i] = calibration_file[i].replace('$fx$', str(calib_params.fx))
        calibration_file[i] = calibration_file[i].replace('$fy$', str(calib_params.fy))
        calibration_file[i] = calibration_file[i].replace('$cx$', str(calib_params.cx))
        calibration_file[i] = calibration_file[i].replace('$cy$', str(calib_params.cy))
        calibration_file[i] = calibration_file[i].replace('$bf$', str(calib_params.bf))

    calibration_path = left_dir + sequence + '.yaml'

    open(calibration_path, 'w').writelines(calibration_file)

    call([model_path + 'stereo_kitti', vocab_path, calibration_path, left_dir[:-1], right_dir[:-1], save_path])



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

            calibration_params = CalibrationParameters(calibration_dir + sequence + '.txt')


            print('computing poses..')
            compute_poses_orb(sequence, calibration_params, config.dir('orb_pose_savedir') + sequence + '/', left_dir, right_dir, config.dir('orbslam_modeldir'),
                                config.dir('orbslam_vocab_dir'))
            print('done.')



