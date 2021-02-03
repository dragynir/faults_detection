import pandas as pd
import numpy as np
import os
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries, watershed


class F3Estimator(object):

    def __init__(self, data_path, cube_shape, metrics_logger, mask_dilate=2):
        self.data_path = data_path
        self.metrics_logger = metrics_logger
        self.cube_shape = cube_shape
        self.mask_dilate = mask_dilate
        self.faults_map, self.bbox_with_faults = self.create_annots()
        

    def create_annots(self):
        ANNOTATIONS = os.path.join(self.data_path + 'Lapteva_faults.csv')
        FAULTS_AREAS = os.path.join(self.data_path, 'faults_areas_clean.csv')
        # CUBE_SHAPE = (651, 951, 462)

        df = pd.read_csv(ANNOTATIONS, dtype={'JX': int, 'JY': int, 'JZ': int})
        df['Tag'].nunique()

        # добавляем для каждой точки локацию
        #  y_from, y_till   z_from, z_till 
        df['YFrom'] = df['Title'].str.split('(').str[1].str.split(',').str[0].astype(np.int32)
        df['YTill'] = df['Title'].str.split('(').str[1].str.split(',').str[1].str.split(')').str[0].astype(np.int32)

        df['ZFrom'] = df['Title'].str.split('(').str[2].str.split(',').str[0].astype(np.int32)
        df['ZTill'] = df['Title'].str.split('(').str[2].str.split(',').str[1].str.split(')').str[0].astype(np.int32)


        print(f'Cube shape: {self.cube_shape}')
        print("Image size nx={}, ny={}, nz={}".format(*self.cube_shape))
        print('No of Faults pixels={}, No of Fault Lines={}'.format(len(df), df['Tag'].nunique()))
        print("Faults location x_min={}, y_min={}, z_min={}".format(*df[['XX', 'YY', 'ZZ']].min()))
        print("Faults location x_max={}, y_max={}, z_max={}".format(*df[['XX', 'YY', 'ZZ']].max()))

        faults_map = np.zeros(self.cube_shape)
        faults_map[df['JX'], df['JY'], df['JZ']] = 1
        locations_bbox = pd.read_csv(FAULTS_AREAS)

        bbox_with_faults = []

        for i, sample in locations_bbox.iterrows():
            if int(sample['region_count']) == 0:
                continue

            slice_index = int(sample['filename'].split('_')[1].split('.')[0])

            bbox = eval(sample['region_shape_attributes'])
            x = int(bbox['x'])
            y = int(bbox['y'])
            width = int(bbox['width'])
            height = int(bbox['height'])
            bbox_with_faults.append((slice_index, x, y, x + width, y + height))
        
        return faults_map, bbox_with_faults

    
    def dilate_mask(self, mask):
        kernel = np.ones((self.mask_dilate,self.mask_dilate), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        return dilation

    def apply_watershed(self, prob, min_tr, max_tr):
        
        thresh = (prob >= max_tr).astype(np.uint8)
        unknown = ((np.logical_and(prob > min_tr, prob < max_tr))).astype(np.uint8)

        ret, markers = cv2.connectedComponents(thresh)

        # background is not 0, but 1
        markers = markers+1

        # mark the region of unknown with zero
        markers[unknown==1] = 0

        img = np.repeat((prob * 255).astype(np.uint8)[..., None], 3, axis=-1)

        watershed = cv2.watershed(img, markers)

        return (watershed > 1).astype(np.float32)

    def estimate(self, cube, pred_mask, use_watershed=False, min_tr=0.2, max_tr=0.6, mid_tr=-1): 
        slice_shape = (462, 951)

        for iteration, sample in enumerate(self.bbox_with_faults):
            i = sample[0] # inline 
            x = sample[1] # xline
            y = sample[2] # time
            end_x = sample[3]
            end_y = sample[4]
            bbox = [i, x, y, end_x, end_y]

            if abs(end_x - slice_shape[0]) < 10:
                end_x = slice_shape[0]
            if abs(end_y - slice_shape[1]) < 10:
                end_y = slice_shape[1]

            seismic = cube[i][x:end_x, y:end_y].T
            true_slice = self.dilate_mask(self.faults_map[i][x:end_x, y:end_y].T)
            pred_slice = pred_mask[i][x:end_x, y:end_y].T

            if use_watershed:
                pred_slice = self.apply_watershed(pred_slice, min_tr, max_tr)
            elif mid_tr != -1:
                pred_slice[pred_slice >= mid_tr] = 1.0
                pred_slice[pred_slice < mid_tr] = 0.0

            self.metrics_logger.log(pred_slice, true_slice, seismic, bbox, iteration)



class LUKEstimator(object):
    def __init__(self, data_path, cube_shape, metrics_logger, mask_dilate):
        self.data_path = data_path
        self.metrics_logger = metrics_logger
        self.mask_dilate = mask_dilate
        self.cube_shape = cube_shape
        self.faults_map, self.bbox_with_faults = self.create_annots()

    def read_annotations(self):
        # predefined constants for annotation: 'faults_planes.dat'
        x_start = 2170
        x_end = 2550
        y_start = 19
        y_end = 990
        z_start = 50
        z_end = 3050

        x_coor = []
        y_coor = []
        z_coor = []
        fault_end = []
        prev_tag = -1

        with open(os.path.join(self.data_path, 'faults_planes.dat'), 'r') as planes:
            for l in planes:
                _, x, y, z, tag, ___ = l.split()
                x_coor.append(int(x) - x_start)
                y_coor.append(int(y) - y_start)
                z_coor.append(  (int(z.split('.')[0]) - z_start) // 2)
                tag = int(tag)
                if prev_tag != -1 and prev_tag != tag:
                    fault_end.append(1)
                else:
                    fault_end.append(0)
                prev_tag = tag

        d = {'x': x_coor, 'y': y_coor, 'z': z_coor, 'faults_end': fault_end}
        df = pd.DataFrame(data=d)

        return df


    def fill_mask(self, df, cube_shape):
        new = df.loc[~df['x'].isin([406])]
        new = new.loc[~new['x'].isin([396])]
        new = new.loc[~new['x'].isin([386])]
        new['x'].max(), new['y'].max(), new['z'].max()

        mask_sgy = np.zeros(cube_shape)
        mask_sgy[new['x'], new['y'], new['z']] = 1

        prev_x, prev_y, prev_z = -1, -1, -1

        for i in range(len(df['x'])):
            if df['x'][i]<=381:
                x = df['x'][i]
                y = df['y'][i]
                z = df['z'][i]
                if prev_x != -1 and df['faults_end'][i] != 1:
                    mask_sgy[x] = cv2.line(mask_sgy[x], (prev_z, prev_y), (z, y), (1), 1)
                    
                prev_x, prev_y, prev_z = x, y, z

        return mask_sgy

   
    def find_test_samples_rects(self, mask_sgy):
        indexes_with_faults = []
        for i in range(len(mask_sgy)):
            if np.sum(mask_sgy[i]) > 0:
                indexes_with_faults.append(i)


        print(f'Slices xline with faults: {len(indexes_with_faults)}')


        slice_fault_rects = dict()
        test_samples_count = 0
        for slice_index in indexes_with_faults:
            mask = mask_sgy[slice_index]
            mask = np.array(mask * 255, dtype = np.uint8)
            contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice_offset_x, slice_offset_y = 32, 16
            rect_mask = np.zeros(mask.shape)
            for c in contours:
                y, x, h, w = cv2.boundingRect(c)
                x_start, y_start = max(0, x - slice_offset_x), max(0, y - slice_offset_y)
                x_end, y_end = min(mask.shape[0], x_start + w + slice_offset_x * 2), min(mask.shape[1], y_start +  h + slice_offset_y * 2)
                cv2.rectangle(rect_mask, (y_start, x_start), (y_end, x_end), (1,), cv2.FILLED) # 1 cv2.FILLED thickness=2, 

            mask_filled = np.array(rect_mask * 255, dtype = np.uint8)

            contours, hier = cv2.findContours(mask_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice_offset_x, slice_offset_y = 32, 16
            faults_rects = []

            
            for c in contours:
                y, x, h, w = cv2.boundingRect(c)
                x_start, y_start = x, y
                x_end, y_end = min(mask.shape[0], x_start + w), min(mask.shape[1], y_start +  h)
                faults_rects.append((x_start, y_start, x_end, y_end))
            slice_fault_rects[slice_index] = faults_rects
            test_samples_count+=len(faults_rects)
        
        print(f'Test samples count: {test_samples_count}')
        
        return slice_fault_rects


    
    def create_annots(self):

        df = self.read_annotations()
        faults_map = self.fill_mask(df, self.cube_shape)
        # test_samples_rects
        # Словарь: ключ - индекс xline (x)
        #          значение - bbox с координатами [x_start, y_start, x_end, y_end]
        bbox_with_faults = self.find_test_samples_rects(faults_map)
    
        return faults_map, bbox_with_faults


    def dilate_mask(self, mask):
        kernel = np.ones((self.mask_dilate,self.mask_dilate), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        return dilation

    
    def estimate(self, cube, pred_mask):        
        iteration = 0
        for iline, bboxes in self.bbox_with_faults.items():
            for bbox in bboxes:
                x_start, y_start, x_end, y_end = bbox
                true_slice = self.dilate_mask(self.faults_map[iline, x_start:x_end, y_start:y_end].T)
                pred_slice = pred_mask[iline, x_start:x_end, y_start:y_end].T
                seismic = cube[iline, x_start:x_end, y_start:y_end].T
                self.metrics_logger.log(pred_slice, true_slice, seismic, [iline] + list(bbox), iteration)
                iteration+=1
