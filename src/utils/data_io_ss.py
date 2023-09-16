import json
import logging
from tqdm import tqdm
import pickle
from pathlib import Path
import copy
import librosa
from joblib import Parallel, delayed
import sys
import torch
import torch.nn.functional as F
import numpy as np
import random
import speechbrain as sb
import speechbrain.dataio.dataio
import speechbrain.dataio.dataset
import speechbrain.dataio.encoder
import speechbrain.utils.data_pipeline
from utils.data_io_utils import get_label_encoder4
import os
import psutil

# from utils.get_spectros import SFFspectrogram, ZTWspectrogram_


logger = logging.getLogger(__name__)

output_keys = [
    'id',
    'sr',    
    'duration',  # total duration of an audio clip
    'wav_path',  # for ADS dection in model
    'wav',  # waveform
    # 'aug_wav',
    'feats',  # Mel feature
    # 'aug_feats',
    'domainFeatures',  # domain features
    'phn_mode_encoded', # phonation modes
    'flvl_phn_mode_encoded',
    'seg_on_list',
    'seg_off_list',
]

def prepare_datasets(hparams):

    logger.info('Preparing datasets.')
    computed_dataset_dir = Path(hparams['prepare']['computed_dataset_dir'])

    # check if pre-computed datasets exist
    set_names = ['train', 'valid', 'test']
    to_prepare = False
    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        if not pkl_path.exists():  # prepare computed dataset
            to_prepare = True

    # prepare dataset or load pre-computed datasets
    computed_datasets = []
    if to_prepare:
        logger.info('One or more computed datasets do not exist, start preparing them.')

        # prepare datasets
        datasets = data_io_prep(hparams)  # <---------------------------------

        # save datasets
        for set_name, dataset in zip(set_names, datasets):  # prepare dataset for train, valid, and test sets
            pkl_path = computed_dataset_dir / f'{set_name}.pkl'
            pkl_path.parent.mkdir(exist_ok=True)
            computed_dataset_dict = {}
            pbar = tqdm(dataset)
            pbar.set_description('computing dataset')
            for data_sample in pbar:  # for each data sample
                utt_id = data_sample['id']  # get ID
                print(u'Mem: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                data_sample_dict = {}  # data sample content as a dictionary
                for key in output_keys:
                    if key != 'id':
                        data_sample_dict[key] = copy.deepcopy(data_sample[key])  # use deep copy to prevent errors
                computed_dataset_dict[utt_id] = data_sample_dict

            def compute(idx):
                data_sample = dataset[idx]
                utt_id = data_sample['id']  # get ID
                data_sample_dict = {}  # data sample content as a dictionary
                for key in output_keys:
                    if key != 'id':
                        data_sample_dict[key] = copy.deepcopy(data_sample[key])  # use deep copy to prevent errors
                return utt_id, data_sample_dict
            compute_datasets_result_list = Parallel(n_jobs=2)(delayed(compute)(idx) for idx in tqdm(range(len(dataset))))

            computed_dataset_dict = {}
            for utt_id, data_sample_dict in compute_datasets_result_list:
                computed_dataset_dict[utt_id] = data_sample_dict

            # save the computed dataset
            with open(pkl_path, 'wb') as f:
                pickle.dump(computed_dataset_dict, f)
    else:
        logger.info('Load pre-computed datasets.')

    # load dataset
    for set_name in set_names:
        pkl_path = computed_dataset_dir / f'{set_name}.pkl'
        with open(pkl_path, 'rb') as f:
            computed_dataset_dict = pickle.load(f)
            
        computed_dataset = sb.dataio.dataset.DynamicItemDataset(computed_dataset_dict, output_keys=output_keys)
        computed_datasets.append(computed_dataset)

# ---------------label encoder-----------------
    label_encoder = get_label_encoder4(hparams)
    # save label_encoder
    label_encoder_path = computed_dataset_dir / 'label_encoder.txt'
    label_encoder.save(label_encoder_path)
# ---------------label encoder-----------------
    

    return computed_datasets, label_encoder


def data_io_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains "train"  "valid" and "test" that correspond
        to the appropriate DynamicItemDataset object.
    """
    # datasets initialization: get data from json
    def dataset_prep(hparams, set_name):
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams['prepare'][f'{set_name}_json_path'],
        )

        if hparams['sorting'] in ['ascending', 'descending']:
            reverse = True if hparams['sorting'] == 'descending' else False
            dataset = dataset.filtered_sorted(sort_key='duration', reverse=reverse)
            hparams['train_dataloader_opts']['shuffle'] = False
        return dataset

    set_names = ['train', 'valid', 'test']
    train_dataset = dataset_prep(hparams, 'train')
    valid_dataset = dataset_prep(hparams, 'valid')
    test_dataset = dataset_prep(hparams, 'test')
    datasets = [train_dataset, valid_dataset, test_dataset]
    label_encoder = get_label_encoder4(hparams)

##--------------------------------
    # audio pipelines
    @speechbrain.utils.data_pipeline.takes('wav_path')
    @speechbrain.utils.data_pipeline.provides('wav', 'feats', 'domainFeatures')  # 'aug_wav', 'aug_feats', 
    def audio_pipeline(wav_path):

        # use librosa instead of sb to get the correct sample rate
        sr = hparams['sample_rate']
        wav, _ = librosa.load(wav_path, sr=sr)
        # print('wav', wav.shape)
        wav = torch.from_numpy(wav)
        yield wav  # wave form

        ## --- domainFeatures ---
        batched_wav = wav.unsqueeze(dim=0)  # add a batch dimension
        domainFeatures = hparams['compute_Dfeatures'](wav_path)
        print(domainFeatures.shape)
        
        ## --- spectrograms ---
        # yield MelSpec  # (NUM_FRAMES, 384)
        # # yield ZTWSpec  # (NUM_FRAMES, 1024)
        # yield SFFSpec  # (NUM_POINTS, 400)

        if hparams['prepare']['sff_data_path'] != 'None':        
            feats = hparams['compute_sff_features'](batched_wav, wav_path)
            # print(feats.dtype)
        else:
            feats = hparams['compute_features'](batched_wav).squeeze(0)

        N_frames = max([domainFeatures.shape[0], feats.shape[0]])
        if domainFeatures.shape[0] != N_frames:
            domainFeatures = F.pad(domainFeatures, (0, 0, 0, int(N_frames-domainFeatures.shape[0])), "constant", 0)

        eps = 1e-14
        if feats.shape[0] != N_frames:
            feats = F.pad(feats, (0, 0, 0, int(N_frames-feats.shape[0])), "constant", eps)

        print('feats', feats.shape)
        print('domainFeatures', domainFeatures.shape)

        assert domainFeatures.shape[0] == feats.shape[0]

        yield feats
        yield domainFeatures  # (NUM_FRAMES, 360)

        # aug_wav = wav
        # batched_aug_wav = aug_wav.unsqueeze(dim=0)
        # # add noise
        # if hparams.get('env_corrupt'):
        #     aug_wav = hparams['env_corrupt'](batched_aug_wav, torch.Tensor([1.0])).squeeze(dim=0)
        #     batched_aug_wav = aug_wav.unsqueeze(dim=0)
        #     print(batched_aug_wav.shape, 'batched_aug_wav')
        # yield aug_wav  # wave form with augmentation/noise

        # print(batched_aug_wav.shape, 'batched_aug_wav1')

        
        # aug_feats = hparams['compute_features'](batched_aug_wav).squeeze(dim=0)
        # print(aug_feats.shape, 'aug_feats')
        # yield aug_feats # feature with augmentation

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @speechbrain.utils.data_pipeline.takes('phonation_list', 'seg_on_Ulist', 'seg_off_Ulist', 'duration', 'wav_path')
    @speechbrain.utils.data_pipeline.provides('flvl_phn_mode_encoded', 'phn_mode_encoded', 'seg_on_list', 'seg_off_list', 'duration', 'wav_path', 'sr')  # 'phn_mode_encoded', 
    def label_pipeline(phonation_list, seg_on_Ulist, seg_off_Ulist, duration, wav_path):

        seg_on_list = seg_on_Ulist
        seg_off_list = seg_off_Ulist
        # calculate frame level phonation modes: return flvl_phn_mode_encoded
        sr = hparams['sample_rate']
        hop_length = hparams['hop_length']/1000*sr  # in sample
        num_frames = round(duration/hop_length)
        # 0: rest, 1: breathy, 2 neutral, 3: pressed
        flvln_phn_mode = np.zeros((num_frames))
        phlvln_phn_mode = np.zeros((len(phonation_list)))

        for i in range(len(phonation_list)):
            phonation = phonation_list[i]
            f_seg_on = int(seg_on_Ulist[i]/hop_length)
            f_seg_off = int(seg_off_Ulist[i]/hop_length)
            # f_duration = int(duration_list[i]/hop_length)
            a = ["0", "B", "N", "P"].index(phonation)
            phlvln_phn_mode[i] = int(a)

            # check gt segon and segoff
            if f_seg_off <= f_seg_on:
                # print('f_seg_off<=f_seg_on')
                continue

            # check f_duration label outlier the num_frames of the segment
            if num_frames >= f_seg_off:
                # f_duration within the segment
                flvln_phn_mode[f_seg_on : f_seg_off] = int(a) * np.ones((f_seg_off-f_seg_on), dtype=int)
            else: 
                # f_duration longer than num_frames of the segment
                flvln_phn_mode[f_seg_on : num_frames] = int(a) * np.ones((num_frames-f_seg_on), dtype=int)

        flvln_phn_mode = flvln_phn_mode.tolist()
        
        def to_phonation(flvln_phn_mode):
            phonationset = ['rest', 'breathy', 'neutral','pressed']
            flvl_phn_mode = [phonationset[[0, 1, 2, 3].index(n)] for n in flvln_phn_mode]
            return flvl_phn_mode

        flvl_phn_mode = to_phonation(flvln_phn_mode)
        flvl_phn_mode_encoded = label_encoder.encode_sequence_torch(flvl_phn_mode)

        # calculate phonation level phonation modes
        phlvl_phn_mode = to_phonation(phlvln_phn_mode)
        phn_mode_encoded = label_encoder.encode_sequence_torch(phlvl_phn_mode)

        return flvl_phn_mode_encoded, phn_mode_encoded, seg_on_list, seg_off_list, duration, wav_path, sr

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

##--------------------------------
    # # set output keys
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_dataset, valid_dataset, test_dataset