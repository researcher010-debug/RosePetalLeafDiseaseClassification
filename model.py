import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torchvision.models import resnet18,ResNet18_Weights

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM=256
HIDDEN_DIM=128

class HybridFeatureExtractor(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.cnn1=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.cnn1.classifier=nn.Identity()
        self.cnn2=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn2.fc=nn.Identity()
        self.project1=nn.Sequential(nn.Linear(1280,FEATURE_DIM),nn.LayerNorm(FEATURE_DIM),nn.ReLU(inplace=True))
        self.project2=nn.Sequential(nn.Linear(512,FEATURE_DIM),nn.LayerNorm(FEATURE_DIM),nn.ReLU(inplace=True))
        self.fusion=nn.Sequential(nn.Linear(FEATURE_DIM*2,FEATURE_DIM),nn.LayerNorm(FEATURE_DIM),nn.ReLU(inplace=True),nn.Dropout(0.3))
    def forward(self,x):
        f1=self.project1(self.cnn1(x))
        f2=self.project2(self.cnn2(x))
        return self.fusion(torch.cat([f1,f2],dim=1))

class DeepQNetwork(nn.Module):
    def __init__(self,num_actions):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(inplace=True),nn.Linear(HIDDEN_DIM,HIDDEN_DIM),nn.ReLU(inplace=True),nn.Linear(HIDDEN_DIM,num_actions))
    def forward(self,state):
        return self.network(state)

class DeepQCNNet(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super().__init__()
        self.features=HybridFeatureExtractor(pretrained)
        self.classifier=nn.Sequential(nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(HIDDEN_DIM,num_classes))
        self.q_network=DeepQNetwork(num_classes)
    def forward(self,x,return_features=False):
        features=self.features(x)
        logits=self.classifier(features)
        q_values=self.q_network(features)
        logits=logits+q_values
        if return_features:
            return logits,features,q_values
        return logits
    def predict(self,x):
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x),dim=1)

def build_model(num_classes,pretrained=True):
    return DeepQCNNet(num_classes,pretrained)