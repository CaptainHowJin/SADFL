#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class LoanDataset(Dataset):

    def __init__(self, data_path, train=True, transform=None, test_size=0.2, random_state=42):
        self.transform = transform
        
        # Load and preprocess the data
        self.data, self.targets = self._load_and_preprocess_data(data_path, test_size, random_state, train)
        
    def _load_and_preprocess_data(self, data_path, test_size, random_state, train):
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Basic preprocessing - select relevant features and handle missing values
        # You can modify these features based on your specific loan dataset
        features_to_use = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'issue_d', 'purpose', 'title', 'zip_code', 'addr_state', 'dti',
            'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
            'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
            'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
            'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
            'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
            'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt',
            'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
            'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
            'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
            'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m'
        ]
        
        # Filter features that exist in the dataset
        available_features = [f for f in features_to_use if f in df.columns]
        
        # Use a simpler set of features for this example
        # You can expand this based on your specific dataset
        simple_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc']
        available_simple_features = [f for f in simple_features if f in df.columns]
        
        if len(available_simple_features) == 0:
            # If none of the expected features are available, use the first few numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            available_simple_features = numeric_columns[:min(10, len(numeric_columns))]
        
        # Select features and target
        if 'loan_status' in df.columns:
            target_column = 'loan_status'
        elif 'default' in df.columns:
            target_column = 'default'
        else:
            # If no obvious target column, create a binary target based on some criteria
            # This is just an example - you should modify based on your dataset
            target_column = df.columns[-1]  # Use the last column as target
        
        X = df[available_simple_features].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode categorical target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            # For binary classification, convert to 0/1
            if len(np.unique(y)) == 2:
                y = np.array(y == 1, dtype=int)
            else:
                # For multi-class, keep as is
                pass
        
        # Ensure y is numpy array
        y = np.array(y)
        
        # Convert to binary classification if needed (default prediction)
        y_unique = np.unique(y)
        if len(y_unique) > 2:
            # Convert to binary: 0 for good loans, 1 for bad loans
            # You may need to adjust this logic based on your specific target variable
            y = np.where(y > 0, 1, 0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if train:
            return torch.FloatTensor(X_train), torch.LongTensor(y_train)
        else:
            return torch.FloatTensor(X_test), torch.LongTensor(y_test)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

def load_loan_dataset(data_path, test_size=0.2, random_state=42):
    train_dataset = LoanDataset(data_path, train=True, test_size=test_size, random_state=random_state)
    test_dataset = LoanDataset(data_path, train=False, test_size=test_size, random_state=random_state)
    
    return train_dataset, test_dataset

def loan_iid(dataset, num_users):
    """
    Sample I.I.D. client data from loan dataset
    :param dataset: loan dataset
    :param num_users: number of users
    :return: dict of data indices for each user
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def loan_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from loan dataset
    :param dataset: loan dataset
    :param num_users: number of users
    :return: dict of data indices for each user
    """
    num_shards, num_imgs = 200, int(len(dataset) / 200)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    
    # Sort labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users
