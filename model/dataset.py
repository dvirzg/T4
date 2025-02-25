import pickle
import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader,TensorDataset

demos = ['agegroup', 'heightgroup','weightgroup', 'gender']


def load_and_process_data(args, device, logging, dataset_name):
    cached_features_file = os.path.join(args.data_dir, 'cached_{}'.format(dataset_name))
    if os.path.exists(cached_features_file) and args.load_cache:
        logging.info('loading cached dataset: {}'.format(dataset_name))
        features, dataset = pickle.load(file=open(cached_features_file, 'rb'))
        return features, dataset
    else:
        logging.info('preprocessing dataset: {}'.format(dataset_name))
        # data: {pat_id: {x_1: [v1, v2,...], x_2: [v1, v2,...]}}
        try:
            data = pickle.load(open(os.path.join(args.data_dir, dataset_name + '.pkl'), 'rb'))
        except FileNotFoundError:
            logging.info('Using synthetic data instead')
            data = pickle.load(open(os.path.join(args.data_dir, 'synthetic_full.pkl'), 'rb'))
            # Convert synthetic data format to expected format
            converted_data = {}
            for pid, datum in data.items():
                converted_datum = {}
                # Use temporal features as main features
                temporal_features = datum['x'].transpose()  # Convert to [timesteps, features]
                for i in range(temporal_features.shape[1]):  # For each feature
                    converted_datum[f'feature_{i}'] = temporal_features[:, i].tolist()
                
                # Extract static features (demographics) from x_static
                x_static = datum['x_static']
                # Map the static features to demographic fields
                converted_datum['agegroup'] = float(x_static[0]) if x_static.shape[0] > 0 else 0.0
                converted_datum['heightgroup'] = float(x_static[1]) if x_static.shape[0] > 1 else 0.0
                converted_datum['weightgroup'] = float(x_static[2]) if x_static.shape[0] > 2 else 0.0
                converted_datum['gender'] = float(x_static[3]) if x_static.shape[0] > 3 else 0.0
                
                # Use y[0] as outcome (factual outcome)
                converted_datum['outcome'] = datum['y'][0].tolist()
                # Use treatment sequence
                converted_datum['treatment'] = datum['a'].tolist()
                # Set death to 0 (not available in synthetic data)
                converted_datum['death'] = 0
                converted_data[pid] = converted_datum
            data = converted_data
            
        try:
            features = pickle.load(open(os.path.join(args.data_dir, 'header.pkl'), 'rb'))
        except FileNotFoundError:
            # If no header.pkl, use feature names we created
            features = [f'feature_{i}' for i in range(20)]  # 20 temporal features
            
        X, X_demo, Treatment, Treatment_label, Y, Death, Mask = [], [], [], [], [], [], []

        os.makedirs('stat/', exist_ok=True)
        output=open('stat/{}.csv'.format(dataset_name), 'w')
        for val in features:
            vals = []
            for datum in data.values():
                if val in datum:  # Check if feature exists in datum
                    if isinstance(datum[val], list):
                        vals += datum[val]
                    else:
                        vals.append(datum[val])
            if vals:  # Only process if we have values
                vals = np.array(vals)
                output.write('{},{:.2f},{:.2f}\n'.format(val, np.mean(vals), np.std(vals)))

        for id, datum in data.items():
            x = []
            if 'x' in datum:  # If this is synthetic data in original format
                # Handle synthetic data format
                temporal_features = datum['x'].transpose()  # Convert to [timesteps, features]
                x = temporal_features
                # Extract demographics from x_static
                x_demo = []
                x_static = datum['x_static']
                for i in range(len(demos)):
                    x_demo.append(float(x_static[i]) if i < x_static.shape[0] else 0.0)
                x_demo = np.array(x_demo)
                # Get outcome and treatment
                y = np.array(datum['y'][0])
                treatment = np.array(datum['a'])
            else:
                # Handle regular data format
                for val in features:
                    if val in datum:
                        x.append(datum[val])
                    else:
                        logging.warning(f"Feature {val} not found in datum {id}")
                        x.append(np.zeros_like(datum[features[0]]))  # Use same length as first feature
                x = np.array(x).T
                # Get demographics from converted format
                x_demo = []
                for demo in demos:
                    x_demo.append(datum[demo])
                x_demo = np.array(x_demo)
                # Get outcome and treatment
                y = np.array(datum['outcome'])
                treatment = np.array(datum['treatment'])

            if len(treatment) < 9:
                continue

            x_pad, treatment_pad, treatment_label, y_pad, mask = pad_and_truncate(id, x, y, treatment, args.max_stay, args.pre_window)

            x_pad = torch.tensor(x_pad.astype(np.float32)).unsqueeze(0)
            treatment_pad = torch.tensor(treatment_pad.astype(np.float32)).unsqueeze(0)
            y_pad = torch.tensor(y_pad.astype(np.float32)).unsqueeze(0)
            mask = torch.tensor(mask.astype(np.float32)).unsqueeze(0)
            x_demo = torch.tensor(x_demo.astype(np.float32)).unsqueeze(0)

            X.append(x_pad)
            Treatment.append(treatment_pad)
            Y.append(y_pad)
            X_demo.append(x_demo)
            Death.append(torch.tensor(datum.get('death', 0), dtype=torch.float32).unsqueeze(0))
            Mask.append(mask)
            Treatment_label.append(torch.tensor(treatment_label).unsqueeze(0))

        X = torch.cat(X).to(device)
        X_demo = torch.cat(X_demo).to(device)

        if dataset_name == 'amsterdamdb':
            X_demo_mean = torch.mean(X_demo, dim=0)
            X_demo_std = torch.std(X_demo, dim=0)
            X_demo_norm = (X_demo-X_demo_mean)/X_demo_std
            X_demo[:,:3] = X_demo_norm[:, :3]

        Treatment = torch.cat(Treatment).to(device)
        Y = torch.cat(Y).to(device)

        Y_mean = torch.mean(Y, dim=0)
        Y_std = torch.std(Y, dim=0)
        Y_norm = (Y-Y_mean) / Y_std

        Death = torch.cat(Death).to(device)
        Mask = torch.cat(Mask).to(device)
        Treatment_label = torch.cat(Treatment_label).to(device)

        if args.negative_sample:
            neg_idx=torch.where(Treatment[:, -args.pre_window].sum(-1) == 0)[0]
            pos_idx = torch.where(Treatment[:, -args.pre_window].sum(-1) == 0)[0]
            shuffle_idx = torch.randperm(len(neg_idx))[:len(pos_idx)//5]
            neg_idx = neg_idx[shuffle_idx]
            sample_ids = torch.cat(pos_idx, neg_idx)
            dataset = TensorDataset(X[sample_ids], X_demo[sample_ids],
                                    Treatment[sample_ids], Treatment_label[sample_ids],
                                    Y_norm[sample_ids], Death[sample_ids], Mask[sample_ids])
        else:
            dataset = TensorDataset(X, X_demo, Treatment,Treatment_label, Y_norm, Death, Mask)

        pickle.dump((features, dataset), file=open(os.path.join(args.data_dir, 'cached_{}'.format(dataset_name)), 'wb'))

        return features, dataset


def pad_and_truncate(id, x, y, treatment, max_stay, pre_window):
    x_len = len(x)
    x_obs = x[:x_len-pre_window]
    t_len = len(treatment)
    treatment_obs = treatment[:t_len - pre_window]
    if x_len < max_stay + pre_window:
        # ((top, bottom), (left, right)):
        x_pad = np.pad(x_obs, ((0, max_stay - len(x_obs)) ,(0, 0)), 'constant')
        y_pad = y[-pre_window-1:]
        if max_stay - len(treatment_obs) < 0:
            print()
        treatment_pad = np.pad(treatment_obs, (0, max_stay - len(treatment_obs)), 'constant')
        treatment_pred = treatment[-pre_window:]
        treatment_pad = np.concatenate((treatment_pad, treatment_pred))

        mask = np.zeros_like(x_pad)
        mask[:(x_len-pre_window)] = 1
    else:
        x_pad = x[:max_stay]
        y_pad = y[max_stay-1:max_stay+pre_window]
        treatment_pad = treatment[:max_stay]
        treatment_pred = treatment[max_stay:max_stay+pre_window]
        treatment_pad = np.concatenate((treatment_pad, treatment_pred))
        mask = np.ones_like(x_pad)

    treatment_pre_str = ''.join(str(int(x)) for x in treatment_pred)
    treatment_label = int(treatment_pre_str, 2)
    return x_pad, treatment_pad, treatment_label, y_pad, mask