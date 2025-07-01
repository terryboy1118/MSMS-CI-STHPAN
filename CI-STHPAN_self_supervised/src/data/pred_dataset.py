import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
from fastdtw import fastdtw
from src.data.timefeatures import time_features
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from matplotlib import colors as mcolors
plot_path = '/home/adam/MSMS-CI-STHPAN/nasdaq_all/pic'
os.makedirs(plot_path, exist_ok=True)  # 如果目錄不存在，則創建
from matplotlib import font_manager
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Stock(Dataset):
    def __init__(self, root_path, market_name, tickers_fname,
                 split='train', size=None,
                 features='MS', data_path='stock/2013-01-01',
                 target='Close', scale=True, timeenc=0, freq='d',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.steps = self.pred_len
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split


        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        # read tickers' name
        tickers = np.genfromtxt(
            os.path.join(self.root_path,self.data_path, '..', self.tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(tickers))
        all_label_seg =[]
        all_data_seg = [] 
        eod_data = []
        masks = []
        ground_truth = []
        base_price = []
        data_time_stamp = []
        
        steps = self.steps
        if self.market_name == 'TSE':
            valid_index = 693
            test_index = 924
            trade_dates = 1188
        else:
            valid_index = 756
            test_index = 1008
            trade_dates = 1245
        border1s = [0, valid_index - self.seq_len, test_index - self.seq_len]
        border2s = [valid_index, test_index, trade_dates]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # read tickers' eod data
        for index, ticker in enumerate(tickers):
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path,
                                              self.market_name + '_' + ticker + '_1.csv'))

            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            # print(cols)
            if self.market_name == 'NASDAQ':
                # remove the last day since lots of missing data
                df_raw = df_raw[:-1]
            
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
            data = df_data.values

            '''
            Wn_indicator = 0.04
            order = 4
            df_raw['pct_return'] = df_raw["Close"].pct_change()
            df_raw['pct_return'].fillna(0, inplace=True)
            df_raw["key_indicator_filtered"] = butter_lowpass_filter(df_raw["Close"].to_numpy(),Wn_indicator,order)
               
            if len(df_raw["key_indicator_filtered"]) == len(df_raw):
                # 將 `key_indicator_filtered` 添加到 DataFrame 的最後一列
                df_raw["key_indicator_filtered"] = df_raw["key_indicator_filtered"]
            df_raw['pct_return_filtered'] = df_raw["key_indicator_filtered"].pct_change()
            df_raw['pct_return_filtered'].fillna(0, inplace=True)
           
            coef_list,turning_points,y_pred_list,normalized_coef_list = get_turning_points(df_raw,
                       min_length_limit = 10,
                       merging_threshold = 0.05,
                       merging_dynamic_constraint = 1,
                       max_length_expectation = 30,
                       dynamic_num = 3,
                       labeling_method = 'quantile',
                       )
           
            data_seg, label_seg, index_seg = label(turning_points, normalized_coef_list, df_raw, dynamic_num = 3, labeling_method= 'quantile')
            # output_path = plot_to_file(data=df_raw,tic=None,y_pred_list=y_pred_list,turning_points=turning_points,low=None,high=None,normalized_coef_list=normalized_coef_list,plot_path=plot_path,plot_feather='Close',suffix='_'+ticker,if_color=True,labeling_method='quantile',)           
            all_data_seg.extend(data_seg)  # 將 data_seg 加進去
            all_label_seg.extend(label_seg)
            '''
            #print(all_data_seg)
            #print(all_label_seg)
            #print(all_index_seg)
            data = data[border1:border2]
            
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis = 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            if index == 0:
                print('#single EOD data shape:', data.shape)
                # [股票数*交易日数*5[5-day,10-day,20-day,30-day,Close]]
                eod_data = np.zeros([len(tickers), data.shape[0],data.shape[1]], dtype=np.float32)
                masks = np.ones([len(tickers), data.shape[0]],dtype=np.float32)
                ground_truth = np.zeros([len(tickers), data.shape[0]],dtype=np.float32)
                base_price = np.zeros([len(tickers), data.shape[0]],dtype=np.float32)

            for row in range(data.shape[0]):
                if abs(data[row][-1] + 1234) < 1e-8:
                    masks[index][row] = 0.0
                elif row > steps - 1 and abs(data[row - steps][-1] + 1234) > 1e-8:
                    ground_truth[index][row] = (data[row][-1] - data[row - steps][-1]) / data[row - steps][-1]
                for col in range(data.shape[1]):
                    if abs(data[row][col] + 1234) < 1e-8:
                        data[row][col] = 1.0 # 空值处理
            eod_data[index, :, :] = data
            base_price[index, :] = data[:, -1]
            data_time_stamp.append(data_stamp)
        '''
        interpolated_pct_return_data_seg = np.array(interpolation(all_data_seg))
        tsne_results = TSNE_run_GPU(interpolated_pct_return_data_seg)
        #print(tsne_results)
        print('plotting TSNE')
        TSNE_plot(
        data=tsne_results,
        label_list=all_label_seg,
        title='_nasdaq_tsne',
        folder_name='/home/adam/MSMS-CI-STHPAN/nasdaq_all/pic_tsne'
        )

        '''
        data_stamp = np.array(data_time_stamp)
        print('#eod_data shape:', eod_data.shape)
        print('#masks shape:', masks.shape)
        print('#ground_truth shape:', ground_truth.shape)
        print('#base_price shape:', base_price.shape)
        print('#data_stamp shape:', data_stamp.shape)
        self.eod_data = eod_data
        self.masks = masks
        self.ground_truth = ground_truth
        self.base_price = base_price
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        mask_seq_len = 16

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
               
        seq_x = self.eod_data[:, s_begin:s_end, :]
        seq_y = self.eod_data[:, r_begin:r_end, :]
        seq_x_mark = self.data_stamp[:, s_begin:s_end, :]
        seq_y_mark = self.data_stamp[:, r_begin:r_end, :]
        
        mask_batch = self.masks[:, s_end - mask_seq_len: index + self.seq_len + self.pred_len]
        mask_batch = np.min(mask_batch, axis=1)
        mask_batch = np.expand_dims(mask_batch, axis=1)
        
        price_batch = np.expand_dims(self.base_price[:, index + self.seq_len - 1], axis=1)
        gt_batch = np.expand_dims(self.ground_truth[:, index + self.seq_len + self.pred_len - 1], axis=1)
           
        if self.use_time_features: 
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark, mask_batch, price_batch, gt_batch)
        else: 
            return _torch(seq_x, seq_y, mask_batch, price_batch, gt_batch) 

    def __len__(self):
        return self.eod_data.shape[1] - self.seq_len - self.pred_len + 1


class Dataset_Pred(Dataset):
    def __init__(self, root_path, split='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)


from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, Wn, order):
    b, a = butter(order, Wn, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_turning_points(data_ori,
                       min_length_limit,
                       merging_threshold,
                       merging_dynamic_constraint,
                       max_length_expectation,
                       dynamic_num,
                       labeling_method,
                       ):
    """
    功能：
    1. 根據轉折點（鄰居斜率相反）將數據分段。
    2. 如果段的長度小於 min_length_limit，則合併該段到鄰居段。
    3. 計算每段的斜率。
    4. 在段長度未滿足期望時，根據 metric 和 dynamic_constraint 進行合併。
    """


    data = data_ori.reset_index(drop=True)


    # Step 1: 找到轉折點
    turning_points = find_index_of_turning(data)
    turning_points = [[i] for i in turning_points]  # 每個轉折點作為一個列表
    turning_points_new = [[turning_points[0][0]]]


    # Step 2: 合併段長小於 min_length_limit 的部分
    for i in range(1, len(turning_points) - 1):
        if turning_points[-1][0] - turning_points[i][0] - 1 < min_length_limit:
            for j in range(i, len(turning_points) - 1):
                turning_points_new[-1].extend(turning_points[j])
            break
        elif turning_points[i][0] - turning_points_new[-1][0] >= min_length_limit:
            turning_points_new.append(turning_points[i])
        else:
            turning_points_new[-1].extend(turning_points[i])
    turning_points_new.append(turning_points[-1])
    turning_points = turning_points_new


    # Step 3: 計算初始斜率
    coef_list = []
    normalized_coef_list = []
    y_pred_list = []
    for i in range(len(turning_points) - 1):
        x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
        adj_cp_model = LinearRegression().fit(
            x_seg,
            data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[i + 1][0]]
        )
        y_pred = adj_cp_model.predict(x_seg)
        normalized_coef_list.append(
            100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]]
        )
        coef_list.append(adj_cp_model.coef_)
        y_pred_list.append(y_pred)


    # Step 4: 動態合併
    #if merging_dynamic_constraint != float('inf'):
        #print('Only merge dynamic <= distance: ', merging_dynamic_constraint)
    merging_round = 0
    if merging_threshold != -1:
        change = True
        while change and merging_round < 20:
            merging_round += 1
            counter = sum(1 for tp in turning_points if tp != [])
            #print(f'merging round: {merging_round}, current number of segments: {counter}')
            change = False


            if merging_dynamic_constraint != float('inf'):
                coef_list = []
                normalized_coef_list = []
                y_pred_list = []
                indexs = []
                turning_points_temp_flat = []
                for i in range(len(turning_points) - 1):
                    if turning_points[i] == []:
                        continue
                    for j in range(i + 1, len(turning_points)):
                        if turning_points[j] != []:
                            next_index = j
                            break
                    x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[next_index][0])]).reshape(-1, 1)
                    adj_cp_model = LinearRegression().fit(
                        x_seg,
                        data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[next_index][0]]
                    )
                    y_pred = adj_cp_model.predict(x_seg)
                    normalized_coef_list.append(
                        100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]]
                    )
                    coef_list.append(adj_cp_model.coef_)
                    y_pred_list.append(y_pred)
                    indexs.append(i)
                    turning_points_temp_flat.append(turning_points[i][0])


                turning_points_temp_flat.append(turning_points[-1][0])
                label, data_seg, label_seg_raw, index_seg = get_label(
                    data=data, turning_points=turning_points_temp_flat, low=None, high=None,
                    normalized_coef_list=normalized_coef_list, tic=None,
                    dynamic_num=dynamic_num, labeling_method='quantile'
                )
                #print(label_seg_raw)
                label_seg = [None for _ in range(len(turning_points) - 1)]
                for i in range(len(indexs)):
                    label_seg[indexs[i]] = label_seg_raw[i]
                   


            distance_list = []
            merge_prohibit_times = 0
            for i in range(len(turning_points) - 1):
                if turning_points[i] == []:
                    continue
                have_next_index = False
                for j in range(i + 1, len(turning_points)):
                    if turning_points[j] != []:
                        next_index = j
                        have_next_index = True
                        break
                if not have_next_index:
                    break
                if turning_points[next_index][0] - turning_points[i][0] < max_length_expectation:
                    left_distance = float('inf')
                    right_distance = float('inf')
                    this_seg = data['key_indicator_filtered'].iloc[
                        turning_points[i][0]:turning_points[next_index][0]
                    ].tolist()
                    if i > 0 and i < len(turning_points) - 1:
                        left_index = None
                        for j in range(i - 1, -1, -1):
                            if turning_points[j] != []:
                                left_index = j
                                break
                        if left_index is not None:
                            left_neighbor = data['key_indicator_filtered'].iloc[
                                turning_points[left_index][0]:turning_points[i][0]
                            ].tolist()
                            left_distance = calculate_distance(left_neighbor, this_seg, merging_round)
                    if i < len(turning_points) - 2:
                        next_index_2 = None
                        for j in range(next_index + 1, len(turning_points) - 1):
                            if turning_points[j] != []:
                                next_index_2 = j
                                break
                        if next_index_2 is not None:
                            right_neighbor = data['key_indicator_filtered'].iloc[
                                turning_points[next_index][0]:turning_points[next_index_2][0]
                            ].tolist()
                            right_distance = calculate_distance(this_seg, right_neighbor, merging_round)
                        else:
                            right_neighbor = data['key_indicator_filtered'].iloc[
                                turning_points[next_index][0]:
                            ].tolist()
                            right_distance = calculate_distance(this_seg, right_neighbor, merging_round)


                    if left_distance != float('inf'):
                        distance_list.append(left_distance)
                    if right_distance != float('inf'):
                        distance_list.append(right_distance)


                    if merging_dynamic_constraint != float('inf'):
                        if next_index < len(label_seg) and right_distance != float('inf') and \
                                merging_dynamic_constraint < abs(label_seg[i] - label_seg[next_index]):
                            if right_distance < merging_threshold:
                                merge_prohibit_times += 1
                            right_distance = float('inf')
                        if i > 0:
                            if left_distance != float('inf') and merging_dynamic_constraint < abs(label_seg[i] - label_seg[left_index]):
                                if left_distance < merging_threshold:
                                    merge_prohibit_times += 1
                                left_distance = float('inf')


                    if min(left_distance, right_distance) < merging_threshold:
                        if left_distance < right_distance:
                            turning_points[left_index] = turning_points[left_index] + turning_points[i]
                        else:
                            turning_points[next_index] = turning_points[i] + turning_points[next_index]
                        change = True
                        turning_points[i] = []
            '''
            print('All distance statistics:')
            print(pd.Series(distance_list).describe())
            print('Your merging_threshold is: ', merging_threshold)
            print(f'Merge prohibit times by merging_dynamic_constraint: {merge_prohibit_times}')
            '''


        turning_points_new = [tp for tp in turning_points if tp != []]
        turning_points = turning_points_new
        '''
        print(f'merging_round in total: {merging_round}, number of segments: {len(turning_points)}')
        print('You may want to tune the merging_threshold and merging_dynamic_constraint to get a better result.')
        '''
        coef_list = []
        normalized_coef_list = []
        y_pred_list = []
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(
                x_seg,
                data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[i + 1][0]]
            )
            y_pred = adj_cp_model.predict(x_seg)
            normalized_coef_list.append(
                100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]]
            )
            coef_list.append(adj_cp_model.coef_)
            y_pred_list.append(y_pred)


    turning_points = [i[0] for i in turning_points]


    return np.asarray(coef_list), np.asarray(turning_points), y_pred_list, normalized_coef_list


def find_index_of_turning( data):
    turning_points = [0]
    data = data.reset_index(drop=True)
    for i in range(data['pct_return_filtered'].size - 1):
        if data['pct_return_filtered'][i] * data['pct_return_filtered'][i + 1] < 0:
            turning_points.append(i + 1)
    if turning_points[-1] != data['pct_return_filtered'].size:
        turning_points.append(data['pct_return_filtered'].size)
    # the last turning point is the end of the data
    return turning_points


def get_label(data, turning_points, low, high, normalized_coef_list, tic, dynamic_num=3, labeling_method='quantile'):
    data = data.reset_index(drop=True)
    data_seg = []


    label = []
    label_seg = []
    index_seg = []


    dynamic_flag = Dynamic_labeler(labeling_method=labeling_method, dynamic_num=dynamic_num, low=low,
                                   high=high,
                                   normalized_coef_list=normalized_coef_list, data=data,
                                   turning_points=turning_points)
    data = data['pct_return_filtered']
    for i in range(len(turning_points) - 1):
        if labeling_method == 'slope' or labeling_method == 'quantile':
            coef = normalized_coef_list[i]
        elif labeling_method == 'DTW':
            coef = i
        flag = dynamic_flag.get(coef)
        label.extend([flag] * (turning_points[i + 1] - turning_points[i]))
        if turning_points[i + 1] - turning_points[i] > 2:
            data_seg.append(data.iloc[turning_points[i]:turning_points[i + 1]].to_list())
            label_seg.append(flag)
            #index_seg.append(tic + '_' + str(i))
    return label, data_seg, label_seg, index_seg






class Dynamic_labeler():
    def __init__(self, labeling_method, dynamic_num, low, high, normalized_coef_list, data, turning_points):
        self.labeling_method = labeling_method
        self.dynamic_num = dynamic_num
        if self.labeling_method == 'slope':
            low, _, high = sorted([low, high, 0])
            self.segments = []
            if high!=low:
                for i in range(1, self.dynamic_num):
                    self.segments.append(low + (high - low) / (dynamic_num-2) * i)
            else:
                self.segments.append(low)
        elif self.labeling_method == 'quantile':
            self.segments = []
            # find the quantile of normalized_coef_list
            for i in range(1, self.dynamic_num):
                self.segments.append(np.quantile(normalized_coef_list, i / dynamic_num))
        elif self.labeling_method == 'DTW':
            # segment the data by turning points
            self.segments = []
            for i in range(len(turning_points) - 1):
                self.segments.append(data['pct_return_filtered'][turning_points[i]:turning_points[i + 1]])
            # run the DTW algorithm to cluster the segments into dynamic_num clusters
            self.labels = self.DTW_clustering(self.segments)
        else:
            raise Exception("Sorry, only slope,quantile and DTW labeling_method are provided for now.")






    def get(self, coef):
        if self.labeling_method == 'DTW':
            return self.labels[coef]
        elif self.labeling_method == 'slope' or self.labeling_method == 'quantile':
            # find the place where coef falls into in segments
            for i in range(self.dynamic_num - 1):
                if coef <= self.segments[i]:
                    flag = i
                    return flag
            return self.dynamic_num - 1
       
def calculate_distance(seg1, seg2, iteration_count, labeling_method='DTW_distance', merging_metric=None):
    # calculate the distance between two segments
    if labeling_method == 'default':
        labeling_method = merging_metric
    if labeling_method == 'DTW_distance':
        # the sampling time increase as the iteration_count increase
        distance = calculate_dtw_distance(seg1, seg2, iteration_count * 3 + 10)
    return distance


def calculate_dtw_distance(seg1, seg2, max_sample_number=3):
    # calculate the dynamic time warping distance between two segments
    # roll the shorter segment on the longer one with step_size, and calculate the mean distance


    # decide the step size and slice length based on the max_calculation_number
    # we want to include every point in the longer segment at least once/ the longer segment slice length is comparable to the shorter segment/ step size is not too small


    if len(seg1) > len(seg2):
        longer = seg1
        shorter = seg2
    else:
        longer = seg2
        shorter = seg1


    step_size = max(1, math.floor((len(longer) - len(shorter)) / max_sample_number))
    # slice_length=int(len(longer)/max_sample_number)
    slice_length = len(shorter)


    distances = []
    for i in range(0, len(longer) - len(shorter), step_size):
        distance, paths = fastdtw(shorter, longer[i:i + slice_length])
        distances.append(distance)
    # normalize the distance by the length of the shorter segment and mean value of the shorter segment
    return np.mean(distances) / (slice_length * np.mean(shorter))






def label(turning_points_dict, norm_coef_list_dict, data_dict,
          dynamic_num, labeling_method, do_TSNE=False, do_DTW=False, interpolation=None, TSNE_run=None, tic_DTW=None, work_dir=os.getcwd()):
    """
    標記數據，返回標記結果的全局結構。
    """


    all_data_seg = []
    all_label_seg = []
    all_index_seg = []  
    # 使用 get_label 函數進行標記
    label, data_seg, label_seg, index_seg = get_label(
        data_dict, turning_points_dict, low=None, high=None, normalized_coef_list=norm_coef_list_dict,tic =None, dynamic_num = 3, labeling_method=labeling_method
    )


    # 更新標記結果
    #print(label)
    #print(data_seg)
    #print(label_seg)
    #print(index_seg)
    data_dict['label'] = label
    all_data_seg.extend(data_seg)
    all_label_seg.extend(label_seg)
    all_index_seg.extend(index_seg)
    #print(data_dict)


    return all_data_seg, all_label_seg, all_index_seg


def plot_to_file(data, tic, y_pred_list, turning_points, low, high, normalized_coef_list,
                 plot_path=None, plot_feather=None, suffix='', if_color=True, labeling_method=None, dynamic_flag=None):
    data = data.reset_index(drop=True)
    # 每個子圖包含最多 100000 個數據點
    segment_length = min(100000, data.shape[0] )
    print('plot_path', plot_path, 'segment_length', segment_length)
    #print(data)
    #print(turning_points)
    #print(normalized_coef_list)
    # 分段處理
    plot_segments = []
    counter = 0
    segments_buffer = [turning_points[0]]
    for index, j in enumerate(range(len(turning_points) - 1)):
        counter += turning_points[j + 1] - turning_points[j]
        segments_buffer.append(turning_points[j + 1])
        if counter > segment_length:
            plot_segments.append(segments_buffer)
            segments_buffer = [turning_points[j + 1]]
            counter = 0
    plot_segments.append(segments_buffer)
    sub_plot_num = len(plot_segments)
    #print(plot_segments)
    #print(sub_plot_num)
    # 設置子圖
    fig, axs = plt.subplots(sub_plot_num, 1, figsize=(50, 15 * sub_plot_num), constrained_layout=True)
    if sub_plot_num == 1:
        axs = [axs]


    # 設置顏色
    if if_color:
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    else:
        colors = ['black' for _ in range(999)]


    # 繪圖
    counter = 0
    for index, ax in enumerate(axs):
        turning_points_seg = plot_segments[index]
        for i in range(len(turning_points_seg) - 1):
            x_seg = np.asarray([j for j in range(turning_points_seg[i], turning_points_seg[i + 1])]).reshape(-1, 1)
            coef = normalized_coef_list[i + counter]
            if labeling_method == 'slope' or labeling_method == 'quantile':
                coef = coef[0]
            elif labeling_method == 'DTW':
                coef = i + counter
            dynamic_flag = Dynamic_labeler(labeling_method=labeling_method, dynamic_num=3, low=low,
                                            high=high,
                                            normalized_coef_list=normalized_coef_list, data=data,
                                            turning_points=turning_points)
            flag = dynamic_flag.get(coef)
            ax.plot(x_seg, data[plot_feather].iloc[turning_points_seg[i]:turning_points_seg[i + 1]],
                    color=colors[flag], label='Market Dynamics ' + str(flag), linewidth=3)
        counter += len(turning_points_seg) - 1


        # 添加圖例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        font = font_manager.FontProperties(weight='bold', style='normal', size=48)
        ax.legend(by_label.values(), by_label.keys(), prop=font)


    # 設置標題
    plt.title(f"Dynamics_of__linear_{labeling_method}_{plot_feather}", fontsize=20)
    fig_path = plot_path + '/' + suffix + '.png'
    fig.savefig(fig_path)
    plt.close(fig)


    return os.path.abspath(fig_path).replace("\\", "/")



def interpolation(data):
    """
    對每個時間序列段進行插值，使其長度統一為最長段的長度。
    補齊部分先以 NaN 插入，再用二次多項式插值補全。
    
    :param data: list of list，每個元素為一段價格序列（長度可能不同）
    :return: list of pd.Series，插值後的價格段，每段長度相同
    """
    max_len = max(len(d) for d in data)

    interpolated_data = []
    for d in data:
        d = list(d)  # 確保是 list
        l = len(d)
        to_fill = max_len - l

        if to_fill > 0:
            interval = max_len // to_fill
            for j in range(to_fill):
                idx = (interval + 1) * j + interval
                d.insert(min(idx, len(d) - 1), float('nan'))

        interpolated_series = pd.Series(d).interpolate(method='polynomial', order=2)
        interpolated_data.append(interpolated_series.tolist())
    #print(interpolated_data)
    return interpolated_data

from sklearn.manifold import TSNE

def TSNE_run_CPU(data_seg):
    interpolated_data = interpolation(data_seg)  # 假設回傳 shape = [N, D]
    
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=1000,
        init='pca',
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    tsne_results = tsne.fit_transform(interpolated_data)
    return tsne_results



'''
from sklearn.manifold import TSNE
def TSNE_run(data_seg):
    interpolated_pct_return_data_seg = np.array(interpolation(data_seg))
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(interpolated_pct_return_data_seg)
'''

def TSNE_plot(data, label_list, title='', folder_name=None):
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        for i in range(len(data) - 1):
            label = label_list[i]
            ax.scatter(data[i][0], data[i][1], color=colors[label], alpha=0.2, label='cluster' + str(label))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('TSNE', fontsize=20)
        plot_path = folder_name
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path + 'TSNE' + title + '.png')
        plt.close(fig)