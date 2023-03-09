"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn.functional as F
import paddle.nn as nn
import paddle
import numpy as np
import paddle.distributed as dist
from args import config

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils


def sigmoid_prob(logits):
    return F.sigmoid(logits - paddle.mean(logits, axis=-1, keepdim=True))


class DenoisingNet(nn.Layer):
    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                paddle.unsqueeze(
                    paddle.zeros_like(
                        input_list[i]).astype('float32'), axis=-1) for _ in range(self.list_size)]
            click_feature[i] = paddle.unsqueeze(
                paddle.ones_like(input_list[i]).astype('float32'), axis=-1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    paddle.concat(
                        click_feature, axis=1)))

        return paddle.concat(output_propensity_list, axis=1)


class DLA(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build DLA')

        self.rank_feature_size = exp_settings['rank_feature_size']

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='adamw',            # Select gradient strategy
        )

        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num'] + \
            exp_settings['negative_num']
        self.feature_size = exp_settings['feature_size']
        self.combine = exp_settings['combine']
        self.change_label = exp_settings['change_label']

        self.feature_id = self.type_idx(config.feature_type)

        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff'] + \
                exp_settings['negative_num']
            self.propensity_model = DenoisingNet(self.rank_list_size)

        # DataParallel
        # initialize parallel environment
        dist.init_parallel_env()
        self.model = encoder_model
        if paddle.device.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = paddle.DataParallel(self.model)

        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.labels_name.append("label{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(
                self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(axis=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = paddle.optimizer.AdamW
        # if self.hparams.grad_strategy == 'sgd':
        #     self.optimizer_func = paddle.optimizer.SGD

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        opt_denoise = self.optimizer_func(
            learning_rate=self.propensity_learning_rate,
            parameters=self.propensity_model.parameters(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
        )
        opt_ranker = self.optimizer_func(
            learning_rate=self.learning_rate,
            parameters=self.model.parameters(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
        )

        opt_denoise.clear_grad()
        opt_ranker.clear_grad()

        self.loss.backward()

        opt_denoise.step()
        opt_ranker.step()

        # print("=============更新之后===========")
        # for name, parms in self.model.named_parameters():
        #     print('-->name:{} | size:{}'.format(name, parms.shape))
        #     print('-->para:', parms)
        #     print('-->stop_gradient:', parms.stop_gradient)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        # for name, parms in self.propensity_model.named_parameters():
        #     print('-->name:{} | size:{}'.format(name, parms.shape))
        #     print('-->para:', parms)
        #     print('-->stop_gradient:', parms.stop_gradient)
        #     print('-->grad_value:', parms.grad)
        #     print("===")

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward)

        """

        # Build model
        self.rank_list_size = self.exp_settings['selection_bias_cutoff'] + \
            self.exp_settings['negative_num']
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # start train
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        features = None
        if self.combine:
            features = input_feed['features'][:, :25]
        q_freq = input_feed['q_freq']

        train_output = self.model(src=src, src_segment=src_segment,
                                  src_padding_mask=src_padding_mask, features=features)

        if self.change_label != 'no':
            all_features = input_feed['features']
            if config.vote:
                self.feature_id = 25
            train_labels = self.process_target(
                self.labels, all_features[:, self.feature_id],
                pos_num=self.exp_settings['max_candidate_num'],
                temperature=config.temperature, change_label=self.change_label,
                delta=config.delta, mode=config.mode)
        else:
            train_labels = self.labels

        train_output = paddle.reshape(
            train_output, shape=[-1, self.max_candidate_num])

        self.propensity_model.train()
        propensity_labels = paddle.transpose(train_labels, perm=[1, 0])
        self.propensity = self.propensity_model(
            propensity_labels)
        with paddle.no_grad():
            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity))
        self.rank_loss = self.loss_func(
            train_output, train_labels, propensity_weights=self.propensity_weights)

        # Compute examination loss
        with paddle.no_grad():
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output))

        self.exam_loss = self.loss_func(
            self.propensity,
            train_labels,
            propensity_weights=self.relevance_weights
        )

        if (self.exp_settings['add_freq'] == "True"):
            self.loss = (self.exp_settings['freq_k']+1) / (self.exp_settings['freq_b']+q_freq) * (
                self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss)
        else:
            self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        self.separate_gradient_update()

        self.clip_grad_value(train_labels, clip_value_min=0, clip_value_max=1)
        self.global_step += 1
        return self.loss.item()

    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        features = None
        if self.combine:
            if self.rank_feature_size == 869:
                features = input_feed['features'][:, :24]
            else:
                features = input_feed['features']
        scores = self.model(src=src, src_segment=src_segment,
                            src_padding_mask=src_padding_mask, features=features)
        return scores

    def state_dict(self):
        return {'model': self.model.state_dict(), 'propensity_model': self.propensity_model.state_dict()}

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (paddle.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (paddle.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = paddle.unbind(
            propensity, axis=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = paddle.stack(pw_list, axis=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights, clip_value_min=0,
                                 clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights
