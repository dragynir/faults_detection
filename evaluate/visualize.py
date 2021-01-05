#@title slice viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets.widgets import *
from ipywidgets import widgets
from IPython.display import display
import torch

plt.style.use('seaborn-white')

class Visualizer(object):

    def __init__(self, metrics_logger):
        self.logger = metrics_logger

    def _autolabel(self, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(round(height, 3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def _create_bar(self, values, names, fig_size):    

        plt.figure(figsize=fig_size)
        x = np.arange(len(names))
        rects = plt.bar(x, values)
        self._autolabel(rects)
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.xticks(x, names)
    
    def show_results(self, fig_size):

        results = self.logger.results()

        avg_metric = []
        names = []
        for metric_name, values in results.items():
            value = torch.mean(torch.tensor(values)).numpy()
            avg_metric.append(value)
            names.append(metric_name)
   
        self._create_bar(avg_metric, names, fig_size)
        
        
    def show_iter_results(self, figsize=(12, 12), alpha=0.5):

        iter_results = self.logger.iter_results()

        for i, iter in iter_results.items():

            fig, axs = plt.subplots(1, 2, figsize=figsize)
            location = iter['location']
            inline = location[0]
            xline = location[1]
            time = location[2]
            title = f'Location: inline: {inline}, xline: {xline}, time: {time}, '

            for k in iter.keys():
                if k not in ['true_mask', 'pred_mask', 'seismic', 'location']:
                    v = iter[k]
                    title+= f'{k}: {v}, '   

            axs[0].set_title(title, loc='left')
            axs[0].imshow(iter['seismic'])
            axs[0].imshow(iter['true_mask'], alpha=0.5)

            axs[1].imshow(iter['seismic'])
            axs[1].imshow(iter['pred_mask'], alpha=0.5)
        
            plt.show()


class SliceViewer():

    '''
        Arguments: 
            traces_volume - np.array with shape (nx, ny, nz)
            labels_volume - np.array with shape (nx, ny, nz)
            figsize - tuple, plot size

    '''

    def __init__(self, traces_volume, labels_volume,
                 figsize=(7, 7), tr_cmap='gray', lb_cmap='Greens'):

        assert traces_volume.shape == labels_volume.shape, 'Shapes must be equal'

        self.tr_volume = traces_volume
        self.lb_volume = labels_volume
        self.figsize = figsize
        self.tr_cmap = tr_cmap
        self.lb_cmap = lb_cmap

        self.nx, self.ny, self.nz = self.tr_volume.shape

        self.widget = self.__create_widget()


    def __create_widget(self):

        x_slider = IntSlider(min=0, max=self.nx - 1, step=1, value=int(self.nx/2))
        y_slider = IntSlider(min=0, max=self.ny - 1, step=1, value=int(self.ny/2))
        z_slider = IntSlider(min=0, max=self.nz - 1, step=1, value=int(self.nz/2))
        alpha = IntSlider(min=0, max=100, step=5, value=70)
        selector = ['x','y','z']
        w = interact(self.__show, x=x_slider, y=y_slider,
                     z=z_slider, alpha=alpha, dim=selector)

        return w

    
    def __get_slice(self, x, y, z, dim):

        if dim == 'x':
            tr = self.tr_volume[x, :, :]
            lb = self.lb_volume[x, :, :]
        elif dim == 'y':
            tr = self.tr_volume[:, y, :]
            lb = self.lb_volume[:, y, :]
        else:
            tr = self.tr_volume[:, :, z]
            lb = self.lb_volume[:, :, z]
        
        return tr, lb


    def __set_labels(self, ax, dim):

        if dim == 'x':
            ax.set_xlabel('y')
            ax.set_ylabel('z')
        elif dim == 'y':
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            ax.set_xlabel('y')
            ax.set_ylabel('x')

    
    def __show(self, x, y, z, alpha, dim):

        tr, lb = self.__get_slice(x, y, z, dim)

        fig, ax = plt.subplots(figsize=self.figsize)
        plt.imshow(tr.T, cmap=self.tr_cmap, aspect='equal')
        plt.imshow(lb.T, alpha=(100-alpha)/100, aspect='equal', cmap=self.lb_cmap)
        self.__set_labels(ax, dim)
        plt.grid()
        plt.show()
        display()

        ax.set_xlabel('x')
        ax.set_ylabel('z')
