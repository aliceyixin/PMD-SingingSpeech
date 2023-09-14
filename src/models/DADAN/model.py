from itertools import count
import logging
import functools
import numpy
import speechbrain as sb
from torch import device
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss
# metrics
from utils.metric_stats.flvl_metric_stats import flvlPMDMetricStats
from utils.metric_stats.phlvl_metric_stats import phlvlPMDMetricStats
from utils.metric_stats.seg_metric_stats import SegMetricStats
from utils.metric_stats.loss_metric_stats import LossMetricStats
from utils.metric_stats.domain_metric_stats import DomainMetricStats
from utils.c2f_postprocess import out_smoothing
from modules.reverselayer import ReverseLayerF, LambdaSheduler
from models.pmd_model import PMDModel
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import os


logger = logging.getLogger(__name__)


class SBModel(PMDModel):

    def on_fit_start(self):
        super(SBModel, self).on_fit_start()
        # load pretrained 'Encoder' and 'Classifier'
        encoder_savepath = os.path.join(self.hparams.savept_path, 'Encoder.pt')
        classifier_savepath = os.path.join(self.hparams.savept_path, 'Classifier.pt')
        self.modules['Encoder'].load_state_dict(torch.load(encoder_savepath))
        self.modules['DEncoder'].load_state_dict(torch.load(encoder_savepath))
        self.modules['Classifier'].load_state_dict(torch.load(classifier_savepath))
        print('Load pretrained model from: ', self.hparams.savept_path)


    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['src_phlvl_stats'] = phlvlPMDMetricStats(self.hparams)
        self.stats_loggers['src_seg_stats'] = SegMetricStats()
        self.stats_loggers['tgt_phlvl_stats'] = phlvlPMDMetricStats(self.hparams)
        self.stats_loggers['tgt_seg_stats'] = SegMetricStats()
        
        # initialize metric stats for losses: D, E_class, E_domain
        for loss_key in self.hparams.metric_keys:
            if loss_key.endswith('_loss'):
                stats_key = loss_key.lower() + '_stats'
                self.stats_loggers[stats_key] = LossMetricStats(loss_key)
       
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # separate selected domain
        ssinger = self.hparams.ssinger
        tsinger = self.hparams.tsinger
        batch_size = feats.shape[0]          

        # separate source input feature
        indices_ssinger = torch.tensor(np.arange(0, batch_size, 2)).to(self.device)
        feats_ssinger = torch.index_select(feats, 0, indices_ssinger)
        # separate target input feature
        indices_tsinger = torch.tensor(np.arange(1, batch_size, 2)).to(self.device)
        feats_tsinger = torch.index_select(feats, 0, indices_tsinger)

        # Encoder input size:(BATCH, F, T)
        feats = torch.cat((feats_ssinger, feats_tsinger), dim=0)
        embedding_z = self.modules['Encoder'](feats)        
        embedding_zd = self.modules['DEncoder'](feats)

        y = self.modules['Classifier'](embedding_z)
        yd = self.modules['DClassifier'](embedding_zd)
        acfunc = nn.Sigmoid()
        yd = acfunc(yd)

        mine_loss = self.modules['MINE'](embedding_z, embedding_zd)   
        if mine_loss < 0:
            mine_loss = torch.tensor(0.).to(self.device)

        predictions = {
            'y': y,  # pred PMD label
            'embedding_z': embedding_z,  # Encoder embedding
            'yd': yd,  # pred domain label
            'embedding_zd': embedding_zd,  # domain Encoder 
            'indices_ssinger': indices_ssinger,
            'indices_tsinger': indices_tsinger,
            'mi_loss': mine_loss,
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):

        feats, feat_lens = batch['feat']
        batch_size = feats.shape[0]

        y = predictions['y']
        embedding_z = predictions['embedding_z']
        embedding_z_ssinger, embedding_z_tsinger = embedding_z.chunk(2, dim=0)

        ##-----compute Domain loss--------
        def DomainADVLoss(self, x, source=True, lamb=1.0):
            x = ReverseLayerF.apply(x, lamb)
            domain_pred = self.modules['Discriminator'](x)
            acfunc = nn.Sigmoid()
            domain_pred = acfunc(domain_pred)
            
            if source:
                domain_label = torch.ones(domain_pred.shape).long()                
            else:
                domain_label = torch.zeros(domain_pred.shape).long()
            # criterion_adv = nn.BCELoss()
            criterion_adv = nn.BCEWithLogitsLoss()
            adv_loss = criterion_adv(domain_pred, domain_label.float().to(self.device))
            return adv_loss, domain_pred


        lamb = 1.0
        self.use_lambda_scheduler = True
        self.lambda_scheduler = LambdaSheduler(gamma=10, max_iter=20)
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()

        source_loss, domain_pred_s = DomainADVLoss(self, embedding_z_ssinger, True, lamb=lamb)
        target_loss, domain_pred_t = DomainADVLoss(self, embedding_z_tsinger, False, lamb=lamb)
        adv_loss = 0.5 * (source_loss + target_loss)

        domain_label_s = torch.ones((domain_pred_s.size(0), domain_pred_s.size(1))).long().to(self.device)
        domain_label_t = torch.zeros((domain_pred_t.size(0), domain_pred_s.size(1))).long().to(self.device)
        domain_label_s_list = domain_label_s.tolist()
        domain_label_t_list = domain_label_t.tolist()

        # map predicted probability to predicted domain label
        domain_pred_s_label = (domain_pred_s > 0.5) * 1
        domain_pred_t_label = (domain_pred_t < 0.5) * 1
        domain_pred_s_list = domain_pred_s_label.tolist()
        domain_pred_t_list = domain_pred_t_label.tolist()

        ##-----compute domain classification loss--------
        yd = predictions['yd']
        yd_gt = torch.concat([torch.ones([int(batch_size/2), yd.shape[1], yd.shape[2]]).long(),
                                torch.zeros([int(batch_size/2), yd.shape[1], yd.shape[2]]).long()])
        criterion_domain = nn.BCEWithLogitsLoss()  #  nn.BCELoss()
        domain_loss = criterion_domain(yd, yd_gt.float().to(self.device))

        ##-----compute PMD loss--------
        # predicted labels
        y_s, y_t =  y.chunk(2, dim=0)
        # ground truth
        phn_mode_label_seqs, phlvl_phn_mode_label_len = batch['phn_mode_encoded']     
        flvl_phn_mode_label_seqs, flvl_phn_mode_label_len = batch['flvl_phn_mode_encoded']

        # source singer gt
        indices_ssinger = predictions['indices_ssinger']
        phn_mode_label_seqs_ssinger = torch.index_select(phn_mode_label_seqs, 0, indices_ssinger)
        phlvl_phn_mode_label_len_ssinger = torch.index_select(phlvl_phn_mode_label_len, 0, indices_ssinger)
        flvl_phn_mode_label_seqs_ssinger = torch.index_select(flvl_phn_mode_label_seqs, 0, indices_ssinger)
        flvl_phn_mode_label_len_ssinger = torch.index_select(flvl_phn_mode_label_len, 0, indices_ssinger)
        
        flvl_phn_mode_pred_ssinger = y_s.permute(0, 2, 1)
        flvl_phn_mode_label_ssinger = flvl_phn_mode_label_seqs_ssinger.data

        # enforce the pred and label have the same num of frames
        if flvl_phn_mode_pred_ssinger.shape[-1] != flvl_phn_mode_label_ssinger.shape[-1]:
            flvl_phn_mode_pred_ssinger = flvl_phn_mode_pred_ssinger[...,:flvl_phn_mode_label_ssinger.shape[-1]]

        criterion_ce = nn.CrossEntropyLoss()  # Frame wise binary cross entropy loss
        ce_loss = criterion_ce(flvl_phn_mode_pred_ssinger, flvl_phn_mode_label_ssinger)  
        pmd_loss = ce_loss

        ##-----compute MI loss--------
        embedding_zd = predictions['embedding_zd']
        mi_loss =  predictions['mi_loss']

        if mi_loss < 0:
            mi_loss = torch.tensor(0.0)

        ##--Total Loss-----------------------
        loss = pmd_loss + adv_loss * self.hparams.lambda_adv + domain_loss * self.hparams.lambda_domain + mi_loss * self.hparams.lambda_mi
        assert not torch.isnan(loss)

        ##-----compute metrics-----
        for domain in ['src', 'tgt']:
            if domain == 'src':
                out = y_s
                indices = torch.tensor(np.arange(0, batch_size, 2)).to(self.device)
            else:
                out = y_t
                indices = torch.tensor(np.arange(1, batch_size, 2)).to(self.device)
            
            phn_mode_label_seqs_domain = torch.index_select(phn_mode_label_seqs, 0, indices)
            phlvl_phn_mode_label_len_domain = torch.index_select(phlvl_phn_mode_label_len, 0, indices)
            flvl_phn_mode_label_seqs_domain = torch.index_select(flvl_phn_mode_label_seqs, 0, indices)
            flvl_phn_mode_label_len_domain = torch.index_select(flvl_phn_mode_label_len, 0, indices)

            ### frame level
            flvl_pred_pmd_lbl_seqs = torch.argmax(out, dim=-1)
            # unpad gt sequences
            flvl_gt_pmd_lbl_seqs_unpad = undo_padding(flvl_phn_mode_label_seqs_domain, flvl_phn_mode_label_len_domain)
            flvl_pred_pmd_lbl_seqs_unpad = undo_padding(flvl_pred_pmd_lbl_seqs, flvl_phn_mode_label_len_domain)

            for i in range(len(flvl_pred_pmd_lbl_seqs_unpad)):
                # if pred label length is larger than gt label length
                if len(flvl_pred_pmd_lbl_seqs_unpad[i]) != len(flvl_gt_pmd_lbl_seqs_unpad[i]):
                    flvl_pred_pmd_lbl_seqs_unpad[i] = flvl_pred_pmd_lbl_seqs_unpad[i][
                                                      :len(flvl_gt_pmd_lbl_seqs_unpad[i])]

            ### phonation level
            phlvl_gt_seg_on_seqs = [batch['seg_on_list'][i] for i in indices]
            phlvl_gt_seg_off_seqs = [batch['seg_off_list'][i] for i in indices]

            phlvl_gt_pmd_lbl_seqs = undo_padding(phn_mode_label_seqs_domain, phlvl_phn_mode_label_len_domain)
            for i in range(len(phlvl_gt_pmd_lbl_seqs)):
                assert len(phlvl_gt_pmd_lbl_seqs[i]) > 0

            ids = [batch['id'][i] for i in indices]
            sr = self.hparams.sample_rate
            hoplen = self.hparams.hop_length  # in ms
            tolerance = self.hparams.smooth_tolerance  # in frames
            phlvl_pred_pmd_lbl_seqs, phlvl_pred_seg_on_seqs, phlvl_pred_seg_off_seqs = out_smoothing(
                flvl_pred_pmd_lbl_seqs_unpad, sr, hoplen, tolerance)

            self.stats_loggers[f'{domain}_phlvl_stats'].append(
                ids=ids,
                sr=sr,
                n_class=self.hparams.n_phonation,
                pred_pmd_lbl_seqs=phlvl_pred_pmd_lbl_seqs,
                gt_pmd_lbl_seqs=phlvl_gt_pmd_lbl_seqs,
                pred_seg_on_seqs=phlvl_pred_seg_on_seqs,
                pred_seg_off_seqs=phlvl_pred_seg_off_seqs,
                gt_seg_on_seqs=phlvl_gt_seg_on_seqs,
                gt_seg_off_seqs=phlvl_gt_seg_off_seqs,
            )

            self.stats_loggers[f'{domain}_seg_stats'].append(
                ids=ids,
                note_on_seqs=phlvl_gt_seg_on_seqs,
                note_off_seqs=phlvl_gt_seg_off_seqs,
                seg_on_seqs=phlvl_pred_seg_on_seqs,
                seg_off_seqs=phlvl_pred_seg_off_seqs,
            )

        self.stats_loggers['pmd_loss_stats'].append(
            pmd_loss,
        )
        self.stats_loggers['adv_loss_stats'].append(
            adv_loss,
        ) 
        self.stats_loggers['domain_loss_stats'].append(
            domain_loss,
        )
        self.stats_loggers['mi_loss_stats'].append(
            mi_loss,
        )   

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=self.hparams.max_key, min_key=self.hparams.min_key)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        super(SBModel, self).on_stage_end(stage, stage_loss, epoch)

        if stage == sb.Stage.TEST:
            output_path = Path(self.hparams.output_dir) / 'test_output' / 'F1class_result.txt'
            self.stats_loggers['tgt_phlvl_stats'].write_sample_results_to_file(output_path, self.label_encoder)
