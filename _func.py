import os
import pandas as pd
import numpy as np
import torch
import dgl
import copy

def get_data(trial_setup_dict, batch_size, vali_split, test_split, as_dataset=True, return_indexes=False, currentdir=None, unique_num_dict=None):
    print("IMPORTING from _func ...")
    print("Data ...")

    # for selecting certain list as training set
    vali_tag = trial_setup_dict.get("vali_selected_tag")
    test_tag = trial_setup_dict.get("test_selected_tag")

    # for selecting certain rct or lig as testing set or training set
    _id_dict = {
        "train": {"rct": [], "lig": []},
        "vali": {"rct": [], "lig": []},
        "test": {"rct": [], "lig": []}
    }
    _EMPTY_id_dict = copy.deepcopy(_id_dict)

    for _x in ["test", "vali", "train"]:
        for _y in ["rct", "lig"]:
            _value = trial_setup_dict.get(f"{_x}_{_y}_id_list")
            if _value:
                _id_dict[_x][_y] = _value

    if not currentdir:
        datadir = os.path.dirname(os.path.realpath(__file__)) + '/../../DATA/'
    else:
        datadir = currentdir + '/../../DATA/'

    # get data from files
    # Ligand data
    LIG_EDGEdataset_filename = trial_setup_dict.get("dataset_edge(LIG)_filename")
    if LIG_EDGEdataset_filename:
        LIG_EDGEdataset = pd.read_csv(datadir + LIG_EDGEdataset_filename)
    else:
        LIG_EDGEdataset = None

    LIG_NODEdataset_filename = trial_setup_dict.get("dataset_node(LIG)_filename")
    if LIG_NODEdataset_filename:
        LIG_NODEdataset = pd.read_csv(datadir + LIG_NODEdataset_filename)
    else:
        LIG_NODEdataset = None

    LIG_EXTRAdataset_filename = trial_setup_dict.get("dataset_extra(LIG)_filename")
    if LIG_EXTRAdataset_filename:
        LIG_EXTRAdataset = pd.read_excel(datadir + LIG_EXTRAdataset_filename)
    else:
        LIG_EXTRAdataset = None

    LIG_data_mode = trial_setup_dict.get("LIG_data_mode", "graphONLY")

    # Reactant data
    RCT_dataset_filename = trial_setup_dict["dataset(RCT)_filename"]
    RCT_dataset = pd.read_excel(datadir + RCT_dataset_filename)

    # Overall data
    RCTxLIG_dataset_filename = trial_setup_dict["dataset(RCTxLIG)_filename"]
    RCTxLIG_dataset = pd.read_excel(datadir + RCTxLIG_dataset_filename)

    if not unique_num_dict:
        unique_num_dict = {'lig': 17, 'rct': 20}

    input_ndata_list = trial_setup_dict["input_ndata_list"]
    input_edata_list = trial_setup_dict["input_edata_list"]
    constant_node_num = trial_setup_dict.get("constant_node_num")

    RCTxLIG_dataset.pop('rct')
    RCTxLIG_dataset.pop('lig')

    # Generate train set
    if vali_tag or test_tag:
        if not vali_tag:
            vali_tag = ["~~NO_SUCH_TAG~~"]

        vali_RCTxLIG_dataset = RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(vali_tag)]
        test_RCTxLIG_dataset = RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(test_tag)]
        train_RCTxLIG_dataset = RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index).drop(test_RCTxLIG_dataset.index)

    elif _id_dict != _EMPTY_id_dict:
        print("selecting specific rct and lig, vali_split and test_split is ignored")

        for _y in ['rct', 'lig']:
            if not _id_dict['train'][_y]:
                _id_dict['train'][_y] = [str(i) for i in range(1, unique_num_dict[_y] + 1)]
                for _id in _id_dict['vali'][_y]:
                    _id_dict['train'][_y].remove(_id)
                for _id in _id_dict['test'][_y]:
                    _id_dict['train'][_y].remove(_id)

        train_tag = [f"{rct_id}_{lig_id}" for rct_id in _id_dict['train']['rct'] for lig_id in _id_dict['train']['lig']]
        train_RCTxLIG_dataset = RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(train_tag)].sample(frac=1, random_state=0)
        testvali_RCTxLIG_dataset = RCTxLIG_dataset.drop(train_RCTxLIG_dataset.index)

        _empty = {col: None for col in RCTxLIG_dataset.columns}
        vali_RCTxLIG_dataset = pd.DataFrame([_empty])

        for _y in ["rct", "lig"]:
            if _id_dict['vali'][_y]:
                _id_list = [int(_x) for _x in _id_dict['vali'][_y]]
                vali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.loc[testvali_RCTxLIG_dataset[f'{_y}_id'].isin(_id_list)]
                testvali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)

        test_RCTxLIG_dataset = pd.DataFrame([_empty])

        for _y in ["rct", "lig"]:
            if _id_dict['test'][_y]:
                _id_list = [int(_x) for _x in _id_dict['test'][_y]]
                test_RCTxLIG_dataset = testvali_RCTxLIG_dataset.loc[testvali_RCTxLIG_dataset[f'{_y}_id'].isin(_id_list)]
                testvali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.drop(test_RCTxLIG_dataset.index)

        if vali_RCTxLIG_dataset['label'].tolist() == [None] and _id_dict['vali'] == {"rct": [], "lig": []}:
            vali_RCTxLIG_dataset = train_RCTxLIG_dataset.sample(frac=vali_split, random_state=0)
            train_RCTxLIG_dataset = train_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)

        if test_RCTxLIG_dataset['label'].tolist() == [None] and _id_dict['test'] == {"rct": [], "lig": []}:
            test_RCTxLIG_dataset = train_RCTxLIG_dataset.sample(frac=test_split, random_state=0)
            train_RCTxLIG_dataset = train_RCTxLIG_dataset.drop(test_RCTxLIG_dataset.index)

        if list(testvali_RCTxLIG_dataset.index):
            print("ERROR, there is something wrong with the valid, test set setting, please change it")
            exit()

    else:
        nontrain_frac = vali_split + test_split
        train_RCTxLIG_dataset = RCTxLIG_dataset.sample(frac=1 - nontrain_frac, random_state=0)
        vali_test_RCTxLIG_dataset = RCTxLIG_dataset.drop(train_RCTxLIG_dataset.index)
        vali_frac = vali_split / (vali_split + test_split)
        vali_RCTxLIG_dataset = vali_test_RCTxLIG_dataset.sample(frac=vali_frac, random_state=0)
        test_RCTxLIG_dataset = vali_test_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)

    train_vali_test_indexes = {
        'train_index': list(train_RCTxLIG_dataset.index),
        'vali_index': list(vali_RCTxLIG_dataset.index),
        'test_index': list(test_RCTxLIG_dataset.index),
        'train_labels': train_RCTxLIG_dataset['label'].tolist(),
        'vali_labels': vali_RCTxLIG_dataset['label'].tolist(),
        'test_labels': test_RCTxLIG_dataset['label'].tolist()
    }
    vali_RCTxLIG_dataset.reset_index(drop=True, inplace=True)
    test_RCTxLIG_dataset.reset_index(drop=True, inplace=True)

    train_tag = train_RCTxLIG_dataset.pop('label')
    train_values = train_RCTxLIG_dataset.pop("k")
    stage1_train_values = train_RCTxLIG_dataset.pop("delta_E")

    vali_tag = vali_RCTxLIG_dataset.pop('label')
    vali_values = vali_RCTxLIG_dataset.pop("k")
    stage1_vali_values = vali_RCTxLIG_dataset.pop("delta_E")

    test_tag = test_RCTxLIG_dataset.pop('label')
    test_values = test_RCTxLIG_dataset.pop("k")
    stage1_test_values = test_RCTxLIG_dataset.pop("delta_E")

    if as_dataset:
        _mode = "train"
    else:
        _mode = "vali_test"
        batch_size = int(1e10)

    def LIGcsv2graph(ndata, ndataname_list, edata
