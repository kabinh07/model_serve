import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TPS_SpatialTransformerNetwork(nn.Module):
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        
        if torch.__version__ > "1.3":
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r

class LocalizationNetwork(nn.Module):
    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        features = self.localization_fc1(features)
        batch_C_prime = self.localization_fc2(features).view(batch_size, self.F, 2)
        return batch_C_prime

class GridGenerator(nn.Module):
    def __init__(self, F, I_r_size):
        super(GridGenerator, self).__init__()
        self.F = F
        self.I_r_height, self.I_r_width = I_r_size
        self.register_buffer("inv_delta_C", torch.zeros(self.F + 3, self.F + 3))
        self.register_buffer("P_hat", torch.zeros(self.I_r_height * self.I_r_width, self.F + 3))

    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(batch_C_prime.device)), 
            dim=1
        )
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime

# --- RESNET ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        return x

class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)

# --- BILSTM ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        if hasattr(self.rnn, 'flatten_parameters'):
            self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

# --- ATTENTION ---
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(input_char.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=False, batch_max_length=25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1
        
        hidden = (torch.zeros(batch_size, self.hidden_size).type_as(batch_H),
                  torch.zeros(batch_size, self.hidden_size).type_as(batch_H))
        
        if is_train:
            targets = text
        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(batch_H.device)

        probs = torch.zeros(batch_size, num_steps, self.num_classes).type_as(batch_H)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
            step_output = self.generator(hidden[0])
            probs[:, i, :] = step_output
            
            if not is_train:
                _, targets = step_output.max(1)
            else:
                targets = text[:, i]

        return probs

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj)) 
        
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha

# --- MAIN MODEL ---
class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, 
            I_size=(32, 200), 
            I_r_size=(32, 200), 
            I_channel_num=input_channel
        )
        
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.SequenceModeling_output = hidden_size
        
        self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)

    def forward(self, input, text, is_train=False):
        input_trans = self.Transformation(input)
        visual_feature = self.FeatureExtraction(input_trans)
        
        # [Batch, Channel, Height, Width] -> [Batch, Width, Channel, Height]
        visual_feature = visual_feature.permute(0, 3, 1, 2)
        
        # Collapse Height to 1 -> [Batch, Width, Channel, 1]
        visual_feature = F.adaptive_avg_pool2d(visual_feature, (None, 1))
        
        # Squeeze Height dim -> [Batch, Width, Channel]
        visual_feature = visual_feature.squeeze(3)
        
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)
        return prediction