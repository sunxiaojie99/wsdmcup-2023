"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.
   paddle
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.cbook import print_cycles

import paddle.nn.functional as F
import paddle
import numpy as np
from abc import ABC, abstractmethod

import baseline_model.utils as utils
# from https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None):
    """
    Replaces NaN, positive infinity, and negative infinity values in input tensor.
    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        nan (float, optional): the value to replace NaNs with. Default is 0.
        posinf (float, optional): if a Number, the value to replace positive infinity values with. If None, positive infinity values are replaced with the greatest finite value representable by input’s dtype. Default is None.
        neginf (float, optional): if a Number, the value to replace negative infinity values with. If None, negative infinity values are replaced with the lowest finite value representable by input’s dtype. Default is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: Results of nan_to_num operation input Tensor ``x``.
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    # posinf_value = paddle.full_like(x, float("+inf"))
    # neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    # assert x.dtype in [paddle.float32, paddle.float64]
    # is_float32 = x.dtype == paddle.float32
    # if posinf is None:
    #     posinf = (
    #         np.finfo(np.float32).max if is_float32 else np.finfo(np.float64).max
    #     )
    # posinf = paddle.full_like(x, posinf)
    # if neginf is None:
    #     neginf = (
    #         np.finfo(np.float32).min if is_float32 else np.finfo(np.float64).min
    #     )
    # neginf = paddle.full_like(x, neginf)
    x = paddle.where(paddle.isnan(x), nan, x)
    # x = paddle.where(x == posinf_value, posinf, x)
    # x = paddle.where(x == neginf_value, neginf, x)
    return x


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = paddle.sum(- labels * F.log_softmax(logits, axis=-1), axis=-1)
    return loss


class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = None  # paddle version

    @abstractmethod
    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a summary containing related information about the step.

        """
        pass

    # feature type=>idx

    def type_idx(self, type):
        mapping = {}
        features = []
        for item in ['title', 'content', 'title_content']:
            for feature in ['tf', 'idf', 'tfidf', 'len', 'bm25', 'JM', 'DIR', 'ABS']:
                features.append('{}_{}'.format(feature, item))
        for idx, feature in enumerate(features):
            mapping[feature] = idx
        return mapping[type]

    def process_target(self, click_label, feature_label, pos_num=10, temperature=0.5, delta=0.1, change_label='max', mode='list'):
        '''click_label: [xx,num_candidates]
           feature_label:[xx]
        '''
        assert change_label in ['max', 'delta']
        query_num = click_label.shape[0]
        num_candidates = click_label.shape[-1]
        feature_label = paddle.to_tensor(
            feature_label.numpy().reshape(-1, num_candidates), dtype='float32')
        neg_num = num_candidates-pos_num
        pos_click = click_label[:, :pos_num]

        # neg_feature: [n_query, neg_num]
        # pos_feature: [n_query, pos_num]
        neg_feature = paddle.zeros([query_num, neg_num])
        pos_feature = feature_label[:, :pos_num]

        last_click = paddle.full(shape=[query_num], fill_value=pos_num-1) - \
            paddle.argmax(pos_click[:, ::-1],
                          axis=1)  # 每个query的last click index(≠ 9)

        for idx, cidx in enumerate(last_click):
            cidx = int(cidx.numpy()[0])

            # 以query全部candidate为平均值
            tmp = cidx+1
            if mode == 'list':
                tmp = pos_num
            cur_delta = paddle.mean(pos_feature[idx, :tmp])*delta
            max_score = paddle.max(pos_feature[idx, :tmp])

            # # delta=>last click之后结果的平均值*delta
            # new_delta = paddle.mean(pos_feature[idx, cidx+1:])*delta
            before_click = pos_click[idx, :cidx+1]
            before_feature = pos_feature[idx, :cidx+1]
            ones = paddle.ones_like(before_click)

            # before click=1=>feature_label+=max
            # before click=0=>feature_label-=new_delta
            if change_label == 'max':
                pos_feature[idx, :cidx+1] = paddle.where(before_click == ones,
                                                         max_score+before_feature, before_feature-cur_delta)
            elif change_label == 'delta':
                pos_feature[idx, :cidx+1] = paddle.where(before_click == ones,
                                                         cur_delta+before_feature, before_feature-cur_delta)
            if mode == 'replace_list':
                pos_feature[idx, :cidx +
                            1] = F.softmax(pos_feature[idx, :cidx+1]/temperature)
                if cidx != pos_num-1:
                    pos_feature[idx, cidx+1:] = 0
        if mode == 'list':
            pos_feature = F.softmax(pos_feature/temperature)
        return paddle.concat([pos_feature, neg_feature], axis=1)

    def create_input_feed(self, input_feed, list_size):
        self.labels = []
        for i in range(list_size):
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = np.transpose(self.labels)
        self.labels = paddle.to_tensor(self.labels).cuda()

    @abstractmethod
    def state_dict(self):
        pass

    def opt_step(self, opt):
        """ Perform an optimization step

        Args:
            opt: Optimization Function to use

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        opt.clear_grad()
        self.loss.backward()
        opt.step()

    def clip_grad_value(self, parameters, clip_value_min=0, clip_value_max=1):
        """Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, paddle.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.clip_(min=clip_value_min, max=clip_value_max)

    def pairwise_cross_entropy_loss(
            self, output, labels, propensity_weights=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (paddle.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (paddle.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (paddle.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (paddle.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = paddle.ones([output.shape[0]])
        loss = softmax_cross_entropy_with_logits(
            logits=output, labels=labels) * propensity_weights
        return paddle.mean(loss)

    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (paddle.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (paddle.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (paddle.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (paddle.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = paddle.ones_like(labels)
        criterion = paddle.nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(output, labels) * propensity_weights
        return paddle.mean(paddle.sum(loss, axis=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (paddle.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (paddle.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (paddle.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (paddle.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = paddle.ones_like(labels)

        loss = None
        sliced_output = paddle.unbind(output, axis=1)
        sliced_label = paddle.unbind(labels, axis=1)
        sliced_propensity = paddle.unbind(propensity_weights, axis=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = paddle.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                cur_pair_loss = - \
                    paddle.exp(
                        sliced_output[i]) / (paddle.exp(sliced_output[i]) + paddle.exp(sliced_output[j]))
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.shape[0]
        return paddle.sum(loss) / batch_size

    def softmax_loss(self, output, labels, click_labels=None, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (paddle.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (paddle.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (paddle.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (paddle.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = paddle.ones_like(
                labels)  # [query_num, max_candidate_num]
        if click_labels is not None:
            propensity_weights = paddle.where(click_labels == paddle.ones_like(
                click_labels), propensity_weights, paddle.ones_like(propensity_weights))
        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            paddle.sum(weighted_labels, axis=1, keepdim=True)
        label_dis = nan_to_num(label_dis)  # 该query没有正点击数据 -> 把nan用0填充
        loss = softmax_cross_entropy_with_logits(
            logits=output, labels=label_dis) * paddle.sum(weighted_labels, axis=1)
        # [query_num]
        return paddle.sum(loss) / paddle.sum(weighted_labels)

    def l2_loss(self, input):
        return paddle.sum(input ** 2)/2
