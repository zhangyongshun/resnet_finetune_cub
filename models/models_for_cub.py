import torch
import torch.nn as nn
#from models.model_resnet_ed import resnet50_ed, resnet101_ed, resnet152_ed
from utils.Config import Config
from utils.weight_init import weight_init_kaiming
#from models.model_resnet_se import se_resnet50, se_resnet101, se_resnet152
from torchvision import models
import os
import numpy as np
#from models.model_resnet import resnet50, resnet101, resnet152

class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)


# class ResNet_self(nn.Module):
#     def __init__(self, pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=50):
#         super(ResNet_self, self).__init__()
#         self.n_class = n_class
#         self.base_model = self._model_choice(model_choice)
#         if pre_trained:
#             model_dict = self.base_model.state_dict()
#             pretrained_dict = torch.load(os.path.join(Config.pretrained_path, 'pretrained_model.pth.tar'), map_location='cpu')['state_dict']
#             key = list(pretrained_dict.keys())[0]
#             new_dict  = {}
#             cnt  = 1
#             for k, v in pretrained_dict.items():
#                 if k[7:] in model_dict and v.size() == model_dict[k[7:]].size():
#                     print('update cnt {}'.format(cnt))
#                     print(k[7:])
#                     cnt += 1
#                     new_dict[k[7:]] = v
#             print(new_dict.keys())
#             model_dict.update(new_dict)
#             self.base_model.load_state_dict(model_dict)
#             #self.base_model.load_state_dict(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-se.pkl'))
#         self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
#         self.base_model.fc.apply(weight_init_kaiming)

#     def load_state_keywise(model, model_path):
#         model_dict = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location='cpu')
#         key = list(pretrained_dict.keys())[0]
#         # 1. filter out unnecessary keys
#         # 1.1 multi-GPU ->CPU
#         if (str(key).startswith('module.')):
#             pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
#                                k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
#         else:
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if
#                                k in model_dict and v.size() == model_dict[k].size()}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 3. load the new state dict
#         model.load_state_dict(model_dict)


#     def forward(self, x):
#         N = x.size(0)
#         assert x.size() == (N, 3, 448, 448)
#         x = self.base_model(x)
#         assert x.size() == (N, self.n_class)
#         return x

#     def _model_choice(self, model_choice):
#         if model_choice == 50:
#             return resnet50()
#         elif model_choice == 101:
#             return resnet101()
#         elif model_choice == 152:
#             return resnet152()




# class ResNet_SE(nn.Module):
#     def __init__(self, pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=50):
#         super(ResNet_SE, self).__init__()
#         self.n_class = n_class
#         self.base_model = self._model_choice(model_choice)
#         if pre_trained:
#             if pre_trained_weight_gpu:
#                 params = np.load(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-se.pkl'))
#                 self.base_model.load_state_dict({i:torch.from_numpy(params[i]) for i in params})
#                 #self.base_model.load_state_dict(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-se.pkl'), map_location=lambda storage, loc: storage)
#             else:
#                 self.base_model.load_state_dict(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-se.pkl'))
#         self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
#         self.base_model.fc.apply(weight_init_kaiming)

#     def forward(self, x):
#         N = x.size(0)
#         assert x.size() == (N, 3, 448, 448)
#         x = self.base_model(x)
#         assert x.size() == (N, self.n_class)
#         return x

#     def _model_choice(self, model_choice):
#         if model_choice == 50:
#             return se_resnet50()
#         elif model_choice == 101:
#             return se_resnet101()
#         elif model_choice == 152:
#             return se_resnet152()



# class ResNet_ED(nn.Module):
#     def __init__(self, pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=50):
#         super(ResNet_ED, self).__init__()
#         self.n_class = n_class
#         self.base_model = self._model_choice(model_choice)
#         if pre_trained:
#             if pre_trained_weight_gpu:
#                 params = np.load(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-ed.pkl'))
#                 self.base_model.load_state_dict({i:torch.from_numpy(params[i]) for i in params}) #= torch.load(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-ed.pkl'),  map_location=lambda storage, loc: storage)
#             else:
#                 self.base_model.load_state_dict(os.path.join(Config.pretrained_path, 'R-'+str(model_choice)+'-ed.pkl'))
#         self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
#         self.base_model.fc.apply(weight_init_kaiming)

#     def forward(self, x):
#         N = x.size(0)
#         assert x.size() == (N, 3, 448, 448)
#         x = self.base_model(x)
#         assert x.size() == (N, self.n_class)
#         return x

#     def _model_choice(self, model_choice):
#         if model_choice == 50:
#             return resnet50_ed()
#         elif model_choice == 101:
#             return resnet101_ed()
#         elif model_choice == 152:
#             return resnet152_ed()


