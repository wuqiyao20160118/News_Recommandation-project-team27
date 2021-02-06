import torch
import torch.nn as nn
from models.encoder import Encoder
from models.attention import AdditiveAttention
import torch.nn.functional as F
import bcolz
from glove import generate_glove_vocab
import pdb


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model.
      Input shape
        - 2D tensor with shape: [batch_size, units].
      Output shape
        - 2D tensor with shape: [batch_size, units].
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
                torch.empty(in_features, in_features))) for i in range(self.layer_num)])
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == 'matrix':
                dot_ = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                print("parameterization should be 'vector' or 'matrix'")
                pass
            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    """
      Arguments
         - task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - use_bias: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X, candidates, labels=None):
        output = X
        if self.use_bias:
            output += self.bias
        output = torch.bmm(X.unsqueeze(1), candidates.permute(0, 2, 1)).squeeze(1)
        #pdb.set_trace()
        # evaluation
        if labels is None:
            if self.task == "binary":
                output = torch.sigmoid(output)
            return output
        return output


class Linear(nn.Module):
    def __init__(self, feature_dim, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(feature_dim, 1).to(device))
        torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, feature):
        logit = feature.matmul(self.weight)
        return logit


class DeepCross(nn.Module):
    """
    Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
        and DCN-M (parameterization='matrix').
    :param hyperParams: hyperParameters defined in config.py
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    """

    def __init__(self, hyperParams, device='cuda:0', dnn_hidden_units=(120, 120)):
        super(DeepCross, self).__init__()
        self.hyperParams = hyperParams
        self.dnn_feature_columns = [hyperParams["model"]["hidden_size"], hyperParams["model"]["hidden_size"]]
        self.dnn_hidden_units = dnn_hidden_units
        self.reg_loss = torch.zeros((1,), device=device)
        self.device = device
        self.embedding_model = self.load_embedding()

        self.cross_num = hyperParams["deep_cross"]["cross_num"]
        self.input_dim = self.compute_input_dim()
        self.dnn = DNN(self.input_dim, dnn_hidden_units,
                       activation=hyperParams["deep_cross"]["dnn_activation"],
                       use_bn=hyperParams["deep_cross"]["dnn_use_bn"],
                       l2_reg=hyperParams["deep_cross"]["l2_reg_dnn"],
                       dropout_rate=hyperParams["deep_cross"]["dnn_dropout"],
                       init_std=hyperParams["deep_cross"]["init_std"], device=device)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.input_dim + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.input_dim

        self.news_encoder_title = Encoder(hyperParams["model"], weight=self.embedding_model)
        self.news_encoder_abstract = Encoder(hyperParams["model"], weight=self.embedding_model)
        self.linear_model = Linear(self.input_dim)
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.crossnet = CrossNet(in_features=self.input_dim,
                                 layer_num=self.cross_num,
                                 parameterization=hyperParams["deep_cross"]["cross_parameterization"], device=device)
        self.regularization_weight = []
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()),
            l2=hyperParams["deep_cross"]["l2_reg_dnn"])
        self.add_regularization_weight(self.dnn_linear.weight, l2=hyperParams["deep_cross"]["l2_reg_linear"])
        self.add_regularization_weight(self.crossnet.kernels, l2=hyperParams["deep_cross"]["l2_reg_cross"])

        self.multi_head = nn.MultiheadAttention(hyperParams["model"]["hidden_size"]*2+dnn_hidden_units[0],
                                                hyperParams["model"]["head_num"], dropout=0.2)
        self.projection = nn.Linear(hyperParams["model"]["hidden_size"]*2+dnn_hidden_units[0],
                                    hyperParams["model"]["hidden_size"]*2+dnn_hidden_units[0])
        self.additive_attention = AdditiveAttention(hyperParams["model"]["hidden_size"]*2+dnn_hidden_units[0],
                                                    hyperParams["model"]["q_size"])

        self.lossFn = nn.CrossEntropyLoss()
        self.out = PredictionLayer(hyperParams["deep_cross"]["task"])
        self.to(device)

    def load_embedding(self):
        """
        Load pre-trained glove embedding model
        :return: pre-trained embedding model
        """
        embedding_size = self.hyperParams["model"]["embedding_size"]
        max_vocab_size = self.hyperParams["max_vocab_size"]
        glove_path = self.hyperParams["glove_path"]
        generate_glove_vocab(glove_path, embedding_size, max_vocab_size)
        embeddings = torch.Tensor(bcolz.open(f'{glove_path}/6B.'+str(embedding_size)+'.dat')[:])
        return embeddings

    def compute_input_dim(self):
        dims = 0
        for feat_num in self.dnn_feature_columns:
            dims += feat_num
        return dims

    def forward(self, clicks_title, clicks_abstract, candidates_title, candidates_abstract, labels=None):
        """
        Forward.
        :param clicks_title: [batch_size, num_clicks, title_len]
        :param candidates_title: [batch_size, num_candidates, title_len]
        :param clicks_abstract: [batch_size, num_clicks, seq_len]
        :param candidates_abstract: [batch_size, num_candidates, seq_len]
        :param labels: labels of the input (None represents evaluation stage.)
        :return: Training: loss, Evaluation: prediction
        """
        batch_size, num_clicks_title, seq_len_title = clicks_title.shape[0], clicks_title.shape[1], clicks_title.shape[2]
        num_candidates = candidates_title.shape[1]
        clicks_title = clicks_title.reshape(-1, seq_len_title)
        candidates_title = candidates_title.reshape(-1, seq_len_title)

        batch_size, num_clicks_abstract, seq_len_abstract = clicks_abstract.shape[0], clicks_abstract.shape[1], clicks_abstract.shape[
            2]
        clicks_abstract = clicks_abstract.reshape(-1, seq_len_abstract)
        candidates_abstract = candidates_abstract.reshape(-1, seq_len_abstract)

        feature_title = self.news_encoder_title(clicks_title)
        feature_abstract = self.news_encoder_abstract(clicks_abstract)
        candidates_title = self.news_encoder_title(candidates_title)
        candidates_abstract = self.news_encoder_abstract(candidates_abstract)
        feature_title = feature_title.reshape(batch_size, num_clicks_title, -1)
        candidates_title = candidates_title.reshape(batch_size, num_candidates, -1)
        feature_abstract = feature_abstract.reshape(batch_size, num_clicks_abstract, -1)
        candidates_abstract = candidates_abstract.reshape(batch_size, num_candidates, -1)

        input = combined_input([feature_title, feature_abstract])
        candidates = combined_input([candidates_title, candidates_abstract])
        input = input.reshape(batch_size*num_clicks_abstract, -1)
        candidates = candidates.reshape((batch_size*num_candidates, -1))

        # # b_logit
        # logit = self.linear_model(input)
        # deep network
        deep_out = self.dnn(input)
        # cross network
        cross_out = self.crossnet(input)
        # stack the features
        stack_out = torch.cat((cross_out, deep_out), dim=-1)

        deep_out_cand = self.dnn(candidates)
        # cross network
        cross_out_cand = self.crossnet(candidates)
        # stack the features
        stack_out_cand = torch.cat((cross_out_cand, deep_out_cand), dim=-1)

        stack_out = stack_out.reshape(batch_size, num_clicks_abstract, -1)
        stack_out_cand = stack_out_cand.reshape(batch_size, num_candidates, -1)

        # # W_logit * x_stack + b_logit
        # logit += self.dnn_linear(stack_out)

        # multi-head attention
        clicks = stack_out.permute(1, 0, 2)
        clicks, _ = self.multi_head(clicks, clicks, clicks)
        clicks = F.dropout(clicks.permute(1, 0, 2), p=0.2)

        # additive attention
        clicks = self.projection(clicks)
        clicks, _ = self.additive_attention(clicks)

        # pdb.set_trace()
        output = self.out(clicks, stack_out_cand, labels=labels)
        if labels is not None:
            # compute loss
            _, labels = labels.max(dim=1)
            output = self.lossFn(output, labels) + self.get_regularization_loss()
        return output

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


def combined_dnn_input(feature_list):
    return torch.flatten(torch.cat(feature_list, dim=-1), start_dim=1)


def combined_input(cand_list):
    return torch.cat(cand_list, dim=-1)


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


if __name__ == "__main__":
    from config import hyperParams
    x = torch.Tensor(1,50, 30).long()
    y = torch.Tensor(1,5, 30).long()
    x2 = torch.Tensor(1,50, 60).long()
    y2 = torch.Tensor(1,5, 60).long()

    model = DeepCross(hyperParams)
    out = model(x, x2, y, y2)
    print(out.shape)
