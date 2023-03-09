import paddle
import paddle.nn as nn
from .base_ranking_model import BaseRankingModel
import baseline_model.utils as utils


class DNN(nn.Layer):

    def __init__(self, projection, feature_size):
        super(DNN, self).__init__()
        self.hparams = utils.hparams.HParams(
            activation_func='elu',
            norm='layer'
        )
        self.feature_size = feature_size
        self.initializer = None
        self.act_func = None
        self.projection = projection
        if feature_size == 101:
            self.hidden_layer_size = [64, 32, 16]
        else:  # 870/793
            self.hidden_layer_size = [512, 256, 128]
        self.output_sizes = self.hidden_layer_size+[1]
        self.layer_norm = None
        self.sequential = nn.Sequential().to(dtype=paddle.float32)

        if projection:
            if feature_size == 869:
                self.projection_tfidf = nn.Linear(3, 17)
                self.projection_bm25 = nn.Linear(3, 21)
                self.projection_JM = nn.Linear(3, 17)
                self.projection_DIR = nn.Linear(3, 17)
                self.projection_ABS = nn.Linear(3, 20)
            if feature_size == 834:
                self.projection_tf = nn.Linear(3, 8)
                self.projection_idf = nn.Linear(3, 8)
                self.projection_tfidf = nn.Linear(3, 8)
                self.projection_doc_len = nn.Linear(3, 8)
                self.projection_bm25 = nn.Linear(3, 8)
                self.projection_JM = nn.Linear(3, 8)
                self.projection_DIR = nn.Linear(3, 8)
                self.projection_ABS = nn.Linear(3, 8)
                self.projection_query_len = nn.Linear(1, 2)

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        # easy to add multi-dense_layer
        for i in range(len(self.output_sizes)):
            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                if self.hparams.norm == 'layer':
                    self.sequential.add_sublayer('layer_norm{}'.format(i),
                                                 nn.LayerNorm(feature_size).to(dtype=paddle.float32))
                else:
                    self.sequential.add_sublayer('batch_norm{}'.format(i),
                                                 nn.BatchNorm2D(feature_size).to(dtype=paddle.float32))
            self.sequential.add_sublayer('linear{}'.format(
                i), nn.Linear(feature_size, self.output_sizes[i]))
            if i != len(self.output_sizes)-1:
                self.sequential.add_sublayer('act{}'.format(i), self.act_func)
            feature_size = self.output_sizes[i]

    def forward(self, input, noisy_params=None, noise_rate=0.05, **kwargs):
        """ Create the DNN model

        Args:
            input: [batch_size,feature_size]
            noisy_params: (dict<parameter_name, paddle.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            [batch_size,1] containing the ranking scores for each instance in input.
        """
        if self.projection:
            # tf-idf [:,2::8] bm25 [4::8] JM [5::8] DIR [6::8] ABS [7::8]
            input = input.cuda()
            if self.feature_size == 834:
                # tmp = paddle.concat((
                #     input[:, 0:25:8],
                #     input[:, 1:25:8],
                #     self.projection_tfidf(input[:, 2:25:8]),
                #     input[:, 3:25:8],
                #     self.projection_bm25(input[:, 4:25:8]),
                #     self.projection_JM(input[:, 5:25:8]),
                #     self.projection_DIR(input[:, 6:25:8]),
                #     self.projection_ABS(input[:, 7:25:8])), axis=1)
                # input = paddle.concat((tmp, input[:, 25:]), axis=1)
                tmp = paddle.concat((
                    self.projection_tf(input[:, 0:24:8]),
                    self.projection_idf(input[:, 1:25:8]),
                    self.projection_tfidf(input[:, 2:25:8]),
                    self.projection_doc_len(input[:, 3:25:8]),
                    self.projection_bm25(input[:, 4:25:8]),
                    self.projection_JM(input[:, 5:25:8]),
                    self.projection_DIR(input[:, 6:25:8]),
                    self.projection_ABS(input[:, 7:25:8]),
                    self.projection_query_len(
                        paddle.unsqueeze(input[:, 24], axis=-1))
                ), axis=1)

                input = paddle.concat((tmp, input[:, 25:]), axis=1)
            elif self.feature_size == 869:
                tmp = paddle.concat((
                    input[:, 0:24:8],
                    input[:, 1:24:8],
                    self.projection_tfidf(input[:, 2:24:8]),
                    input[:, 3:24:8],
                    self.projection_bm25(input[:, 4:24:8]),
                    self.projection_JM(input[:, 5:24:8]),
                    self.projection_DIR(input[:, 6:24:8]),
                    self.projection_ABS(input[:, 7:24:8])), axis=1)
                input = paddle.concat((tmp, input[:, 24:]), axis=1)
        if noisy_params == None:
            output = self.sequential(input)
        else:
            for name, parameter in self.sequential.named_parameters():
                if name in noisy_params:
                    with paddle.no_grad():
                        noise = noisy_params[name]*noise_rate
                        if paddle.device.is_compiled_with_cuda():
                            noise = paddle.to_tensor(noise, place=input.place)
                        parameter += noise
        return output.squeeze(axis=-1)
