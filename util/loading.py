import h5py
import os
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class dl():
    @staticmethod
    def load_h5py(file_path):
        out_dict = {}
        with h5py.File(file_path, 'r') as f:
            attrs = f['attributes']
            out_dict['neuron_name'] = attrs['neuron_name'][:]
            out_dict['neuron_id'] = attrs['neuron_id'][:]
            out_dict['session'] = attrs['session'][:]
            out_dict['fr']         = attrs['fr'][:]
            out_dict['v_x']        = attrs['v_x'][:]
            out_dict['v_y']        = attrs['v_y'][:]
            out_dict['x_pos']      = attrs['x_pos'][:]
            out_dict['y_pos']      = attrs['y_pos'][:]

        return out_dict
    
    @staticmethod
    def filter_neuron(in_dict, filter_neuron):
        names = [n.decode('utf-8').lower() for n in in_dict['neuron_name']]
        mf_ind = [i for i, name in enumerate(names) if filter_neuron in name]
        u_names = sorted(set([ names[i] for i in  mf_ind]))
        mf_id = in_dict['neuron_id'][ mf_ind]
        mf_sess = in_dict['session'][ mf_ind]
        fr_mf     = in_dict['fr'][ mf_ind]
        v_x_mf    = in_dict['v_x'][ mf_ind]
        v_y_mf    = in_dict['v_y'][ mf_ind]
        x_pos_mf  = in_dict['x_pos'][ mf_ind]
        y_pos_mf  = in_dict['y_pos'][ mf_ind]
        aggregated = defaultdict(lambda: {
            'fr': [],
            'v_x': [],
            'v_y': [],
            'x_pos': [],
            'y_pos': []
        })

        for i in range(len(mf_ind)):
            key = (mf_sess[i], mf_id[i])
            aggregated[key]['fr'].append(fr_mf[i])
            aggregated[key]['v_x'].append(v_x_mf[i])
            aggregated[key]['v_y'].append(v_y_mf[i])
            aggregated[key]['x_pos'].append(x_pos_mf[i])
            aggregated[key]['y_pos'].append(y_pos_mf[i])

        ind = 0
        for key, data in aggregated.items():
            ind = ind + 1
            for k in data:
                data[k] = np.concatenate(data[k])

        print('unique names found: ')
        for name in u_names:
            print(name)

        print('total units found: ' + str(ind))

        #print('Number of neurons found: ' + (aggregated.items().shape))


        return aggregated
    
    @staticmethod
    def make_ds(in_dict, key, bias = False):
        neuron_data = in_dict[key]
        if bias:
            X = np.stack([
                neuron_data['v_x'],
                neuron_data['v_y'],
                neuron_data['x_pos'],
                neuron_data['y_pos'],
                np.ones_like(neuron_data['v_x'])  # bias
            ], axis=1)
        else:
            X = np.stack([
                neuron_data['v_x'],
                neuron_data['v_y'],
                neuron_data['x_pos'],
                neuron_data['y_pos'],
            ], axis=1)


        y = neuron_data['fr']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        return X_train, X_test, y_train, y_test