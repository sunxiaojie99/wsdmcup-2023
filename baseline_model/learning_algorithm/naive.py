"""The navie algorithm that directly trains ranking models with clicks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.distributed as dist

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils
from args import config


class NaiveAlgorithm(BaseAlgorithm):
    """The navie algorithm that directly trains ranking models with input labels.

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
        """
        print('Build NaiveAlgorithm')

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='adamw',            # Select gradient strategy
        )
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff'] + \
                exp_settings['negative_num']
        self.feature_size = exp_settings['feature_size']
        self.change_label = exp_settings['change_label']
        self.combine = exp_settings['combine']
        self.loss_func = 'softmax_loss'
        self.mode = config.mode

        # DataParallel
        # initialize parallel environment
        dist.init_parallel_env()
        self.model = encoder_model
        if paddle.device.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = paddle.DataParallel(self.model)

        self.max_candidate_num = exp_settings['max_candidate_num'] + \
            exp_settings['negative_num']
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        if self.mode != 'pair':
            # the labels for the documents (e.g., clicks)
            self.labels_name = []
            self.labels = []  # the labels for the documents (e.g., clicks)
            for i in range(self.max_candidate_num):
                self.labels_name.append("label{0}".format(i))

        self.optimizer_func = paddle.optimizer.AdamW(
            learning_rate=self.learning_rate,
            parameters=self.model.parameters(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
        )

    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        features = None
        if config.combine:
            features = input_feed['features']
        scores = self.model(src=src, src_segment=src_segment,
                            src_padding_mask=src_padding_mask, features=features)
        return scores

    def state_dict(self):
        return {'model': self.model.state_dict()}

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward)

        """
        self.global_step += 1
        self.model.train()
        propensity_weight = None
        propensity_weight = [1., 1.21945965, 1.53417158, 1.70732868, 2.10416102, 2.04434371,
                             2.18222475, 2.39699650, 2.59346581, 2.92519665]
        propensity_weight.extend([0]*self.exp_settings['negative_num'])
        propensity_weight = paddle.to_tensor(
            propensity_weight, dtype=paddle.float32)

        src = input_feed['src']  # [query_num*max_candidate_num, 128]
        # [query_num*max_candidate_num, 128]
        src_segment = input_feed['src_segment']
        # [query_num*max_candidate_num, 128]
        src_padding_mask = input_feed['src_padding_mask']
        features = None
        self.loss = None
        if self.combine:
            features = input_feed['features'][:, :25]
        train_output = self.model(src=src, src_segment=src_segment,
                                  src_padding_mask=src_padding_mask, features=features)
        if self.mode != 'pair':
            self.create_input_feed(input_feed, self.rank_list_size)

            if self.change_label != 'no':
                all_features = input_feed['features']
                idx = self.type_idx(config.feature_type)
                if config.vote:
                    idx = 25
                train_labels = self.process_target(
                    self.labels, all_features[:, idx],
                    pos_num=self.exp_settings['max_candidate_num'],
                    temperature=config.temperature, change_label=self.change_label,
                    delta=config.delta, mode=self.mode)
            else:
                train_labels = self.labels

            train_output = paddle.reshape(
                train_output, shape=[-1, self.max_candidate_num])  # [query_num, max_candidate_num]
            if self.loss_func == 'sigmoid_loss':
                self.loss = self.sigmoid_loss_on_list(
                    train_output, train_labels, self.labels, propensity_weight)
            elif self.loss_func == 'pairwise_loss_on_list':
                self.loss = self.pairwise_loss_on_list(
                    train_output, train_labels, self.labels, propensity_weight)
            else:
                self.loss = self.softmax_loss(
                    train_output, train_labels, self.labels, propensity_weight)
        else:
            train_output = paddle.reshape(
                train_output, shape=[-1, 2])  # pairwise
            train_labels = paddle.stack([paddle.ones(
                [train_output.shape[0]]), paddle.zeros([train_output.shape[0]])], axis=-1)
            self.loss = self.pairwise_cross_entropy_loss(
                train_output, train_labels)

        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func)

        self.clip_grad_value(train_labels, clip_value_min=-1, clip_value_max=1)
        return self.loss.item()
