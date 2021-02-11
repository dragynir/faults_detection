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
import bruges
import numpy as np
# !git clone https://github.com/ashawkey/volumentations.git
from volumentations.volumentations import *
import tensorflow as tf

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



class GaussianPredictor:
    '''
        dt: Sampling interval (ms) for f3
        t_lng: Length of Ricker wavelet in ms
        mode: TestTime augs mode one of avg, max, none

    '''
    def __init__(self, model, norm_func, overlap=12, input_dim=(128, 128, 128), mode='avg', dt=0.004, t_lng=0.082):
        self.dt = dt
        self.model = model
        self.norm_func = norm_func
        self.overlap = overlap
        self.input_dim = input_dim
        self.mode = mode
        self.t_lng = t_lng
        FLIP_INLINE = Compose([Flip(0, p=1)])
        FLIP_XNLINE = Compose([Flip(1, p=1)])
        self.test_time_augs = [FLIP_INLINE, FLIP_XNLINE]
    
    @staticmethod
    def standardise(x):
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        return (x - mean) / std

    @staticmethod
    def scale_min_max(x):
        min_v = tf.math.reduce_min(x)
        max_v = tf.math.reduce_max(x)
        x = (x - min_v) / (max_v - min_v)
        return x.numpy() # 2 * x - 1 # [-1, 1]

    @staticmethod
    def normalize(x):
        x = GaussianPredictor.standardise(x)
        x = GaussianPredictor.scale_min_max(x)
        return x

    def convolve_wavelet(self, target_trace):
        ''' Convolve with a Ricker wavelet '''
        refl = np.reshape(target_trace, [target_trace.shape[0] * target_trace.shape[1], target_trace.shape[2]])
        f = 30
        traces = np.zeros_like(refl)
        wl = bruges.filters.wavelets.ricker(self.t_lng, self.dt, f)
        for i in range(refl.shape[0]):
            traces[i,:] = np.convolve(refl[i,:], wl, mode='same')
        c_traces = np.reshape(traces, target_trace.shape)
        return c_traces

    # set gaussian weights in the overlap bounaries
    def getMask(self, os, input_dim):
        n1, n2, n3 = input_dim
        sc = np.zeros((n1,n2,n3),dtype=np.single)
        sc = sc+1
        sp = np.zeros((os),dtype=np.single)
        sig = os/4
        sig = 0.5/(sig*sig)
        for ks in range(os):
            ds = ks-os+1
            sp[ks] = np.exp(-ds*ds*sig)
        for k1 in range(os):
            for k2 in range(n2):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k1]
                    sc[n1-k1-1][k2][k3]=sp[k1]
        for k1 in range(n1):
            for k2 in range(os):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k2]
                    sc[k1][n3-k2-1][k3]=sp[k2]
        for k1 in range(n1):
            for k2 in range(n2):
                for k3 in range(os):
                    sc[k1][k2][k3]=sp[k3]
                    sc[k1][k2][n3-k3-1]=sp[k3]
        return sc

    def perform_augs(self, x, aug):
        data = {'image': x, 'mask': x}
        aug_data = aug(**data)
        x, _ = aug_data['image'], aug_data['mask']
        return x

    def use_test_time_augs(self, gs, model, mode='avg'):
        samples = [model.predict(gs).squeeze()]
        for i, aug in enumerate(self.test_time_augs):
            gs_aug = self.perform_augs(gs.squeeze(), aug)
            pr = model.predict(gs_aug[None, ..., None]).squeeze()
            pr = self.perform_augs(pr, aug)
            samples.append(pr)
        if mode == 'avg': 
            res = np.average(np.stack(samples, axis=0), axis=0)
        elif mode == 'max':
            res = np.max(np.stack(samples, axis=0), axis=0)
        return res[None, ..., None]
   
    def predict_cube(self, data_sgy):
        m1,m2,m3 = data_sgy.shape
        n1, n2, n3 = self.input_dim
        gx = data_sgy

        os = self.overlap
        c1 = np.round((m1+os)/(n1-os)+0.5)
        c2 = np.round((m2+os)/(n2-os)+0.5)
        c3 = np.round((m3+os)/(n3-os)+0.5)
        c1 = int(c1)
        c2 = int(c2)
        c3 = int(c3)
        p1 = (n1-os)*c1+os
        p2 = (n2-os)*c2+os
        p3 = (n3-os)*c3+os
        gx = np.reshape(gx,(m1,m2,m3))
        gp = np.zeros((p1,p2,p3),dtype=np.single)
        gy = np.zeros((p1,p2,p3),dtype=np.single)
        mk = np.zeros((p1,p2,p3),dtype=np.single)
        gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
        gp[0:m1,0:m2,0:m3]=gx
        sc = self.getMask(os, self.input_dim)


        for k1 in range(c1):
            print(f'Iteration: {k1}, of {c1}')
            for k2 in range(c2):
                for k3 in range(c3):
                    b1 = k1*n1-k1*os
                    e1 = b1+n1
                    b2 = k2*n2-k2*os
                    e2 = b2+n2
                    b3 = k3*n3-k3*os
                    e3 = b3+n3
                    gs[0,:,:,:,0] = gp[b1:e1,b2:e2,b3:e3]
                    gs = self.norm_func(gs)
                    
                    if self.mode == 'none':
                        Y = self.model.predict(gs)
                    else:
                        Y = self.use_test_time_augs(gs, self.model, mode=self.mode)

                    Y = np.array(Y)
                    gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
                    mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
    
        gy = gy/mk
        gy = gy[0:m1,0:m2,0:m3]
        return gy



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