import segyio
import numpy as np
import json
import pandas as pd
import pickle
import os
from tqdm import tqdm_notebook as tqdm
from obspy.io.segy.core import _read_segy as read_segy
from obspy.io.segy.segy import _read_segy
import shutil
import skimage


class CubeLoader(object):
    def __init__(self):
        pass
    
    @staticmethod
    def load_cube(path):
        image_sgy_stream = read_segy(path)
        image_sgy = np.stack(t.data for t in image_sgy_stream.traces)

        print(f'Data shape: {image_sgy.shape}')

        with segyio.open(path) as segyfile:
            segyfile.mmap()
            iline = segyfile.ilines
            cline = segyfile.xline
            sample = segyfile.samples

        x_size, y_size, z_size = len(iline), len(cline), len(sample)
        data_sgy = image_sgy.reshape(x_size, y_size, z_size)

        print(f'Cube shape: {data_sgy.shape}')

        return data_sgy

    @staticmethod
    def load_f3_demo_cube(path):
        image_sgy_stream = read_segy(path)
        cube = np.stack(t.data for t in image_sgy_stream.traces)

        data_sgy = np.zeros((651 * 951, 462))
        data_sgy[:cube.shape[0], :] = cube[:, :462]
        cube = data_sgy.reshape((651, 951, 462))

        print(f'Cube shape: {cube.shape}')
        
        return cube

    @staticmethod
    def get_cube_view(cube, model_input_size=128, overlap=32):

        ilines, xlines, ztime = cube.shape

        pad_ilines = ilines % model_input_size
        pad_xlines = xlines % model_input_size
        pad_ztime = ztime % model_input_size
        cube = np.pad(cube, ((0, pad_ilines), (0, pad_xlines), (0, pad_ztime)))

        step = model_input_size - overlap
        window_shape = [model_input_size, model_input_size, model_input_size]
        window_viewer = skimage.util.view_as_windows(cube, window_shape, step=step)

        return cube, window_viewer

    @staticmethod
    def prepare_cube(path, f3_demo=False, model_input_size=128, overlap=32):
        ''''
            overlap - overlap between prediction segments
            return:
                cube, cube_view
        '''

        if f3_demo:
            cube = CubeLoader.load_f3_demo_cube(path)
        else:
            cube = CubeLoader.load_cube(path)
        
        return CubeLoader.get_cube_view(cube, model_input_size=model_input_size, overlap=overlap)



class Normalization(object):
    
    @staticmethod
    def scale_min_max(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x
    
    @staticmethod
    def standardize(x):
        x = (x - x.mean()) / x.std()
        return x
    @staticmethod
    def custom(x):
        pass


class Predictor(object):
    def __init__(self, normalization=Normalization.scale_min_max, save_path='labels/'):
        self.save_path = save_path
        self.norm = normalization

    def predict_subvolume(self, volume, model):

        volume = self.norm(volume)
        
        y = model(volume[None, ..., None]).numpy()

        return y.squeeze()


    def predict_per_window(self, window_viewer, model):
        
        nx, ny , nz = window_viewer.shape[:3]

        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.mkdir(self.save_path)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    y = self.predict_subvolume(window_viewer[ix, iy, iz], model)
                    path = os.path.join(self.save_path, f'{ix}_{iy}_{iz}.dat')
                    y.flatten().astype('float32').tofile(path, format='%.4f')
    
    def merge_predictions(self, cube, model_input_size=128, overlap=32):

        # TODO добавить возможность усреднять результаты по предсказаниям

        mask = np.zeros(cube.shape)
        step = model_input_size - overlap
        window_shape = [model_input_size, model_input_size, model_input_size]

        for f in os.listdir(self.save_path):
            ix, iy, iz = map(int, f.split('.')[0].split('_'))
            lb = np.fromfile(os.path.join(self.save_path, f), dtype=np.float32)
            lb = np.reshape(lb, window_shape)
            mask[ix * step: (ix) * step + model_input_size,
                        iy * step: (iy) * step + model_input_size, iz * step: (iz) * step + model_input_size] = lb
        
        return mask