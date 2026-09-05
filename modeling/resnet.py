import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import timm

vit_pretrained = timm.create_model('vit_base_patch16_224', pretrained=True)

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
            self,
            block,
            layers,
            ms_class=None,
            ms_layers=[],
            ms_p=0.5,
            ms_a=0.1,
            **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)
        # v = self.global_avgpool(x)
        # return v.view(v.size(0), -1)
        return x

    # def _init_params(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(
    #                m.weight, mode="fan_out", nonlinearity="relu"
    #            )
    #            if m.bias is not None:
    #                nn.init.constant_(m.bias, 0)
    #        elif isinstance(m, nn.BatchNorm2d):
    #            nn.init.constant_(m.weight, 1)
    #            nn.init.constant_(m.bias, 0)
    #        elif isinstance(m, nn.BatchNorm1d):
    #            nn.init.constant_(m.weight, 1)
    #            nn.init.constant_(m.bias, 0)
    #        elif isinstance(m, nn.Linear):
    #            nn.init.normal_(m.weight, 0, 0.01)
    #            if m.bias is not None:
    #                nn.init.constant_(m.bias, 0)
    # def forward(self, x):
    #     f = self.featuremaps(x)
    #     v = self.global_avgpool(f)
    #     return v.view(v.size(0), -1)
    #     #return v


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    # pretrain_dict = torch.load(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


def load_vit_pretrain(vit_model, checkpoint_path=None):
    # Hoặc từ timm
    pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit_model.load_state_dict(pretrained_vit.state_dict(), strict=False)
    return vit_model


class ViTEncoder(nn.Module):
    def __init__(self, input_dim, dim=768, depth=6, heads=8, mlp_dim=2048, dropout=0.1, num_patches=49):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.pos_embedding = None
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x = self.linear_proj(x)
        if self.pos_embedding is None:
            self.pos_embedding = nn.Parameter(torch.randn(1, N + 1, x.size(-1)))

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        return self.norm(x)[:, 0]


class ResNetWithViT(nn.Module):
    def __init__(self, resnet, vit, num_classes):
        super().__init__()
        self.resnet = resnet
        self.vit = vit
        self.fc = nn.Linear(vit.dim, num_classes)
        self._out_features = vit.dim

    def forward(self, x, return_features=False):
        f_map = self.resnet.featuremaps(x)
        B, C, H, W = f_map.shape

        f_seq = f_map.flatten(2).transpose(1, 2)  # [B, H*W, C]

        vit_feat = self.vit(f_seq)  # [B, D]
        return vit_feat

    def out_features(self):
        return self._out_features


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]

"""


def resnet18(num_classes=5, pretrained=True, vit_dim=768, vit_depth=6, vit_heads=8, vit_mlp=2048):
    resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    if pretrained:
        init_pretrained_weights(resnet, model_urls["resnet18"])

    vit = ViTEncoder(input_dim=resnet._out_features, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp)
    vit = load_vit_pretrain(vit)

    model = ResNetWithViT(resnet, vit, num_classes)
    return model


def resnet34(num_classes=5, pretrained=True, vit_dim=768, vit_depth=6, vit_heads=8, vit_mlp=2048):
    resnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])
    if pretrained:
        init_pretrained_weights(resnet, model_urls["resnet34"])

    vit = ViTEncoder(input_dim=resnet._out_features, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp)
    vit = load_vit_pretrain(vit)

    model = ResNetWithViT(resnet, vit, num_classes)
    return model


def resnet50(num_classes=5, pretrained=True, vit_dim=768, vit_depth=6, vit_heads=8, vit_mlp=2048):
    resnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    if pretrained:
        init_pretrained_weights(resnet, model_urls["resnet50"])

    vit = ViTEncoder(input_dim=resnet._out_features, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp)
    vit = load_vit_pretrain(vit)

    model = ResNetWithViT(resnet, vit, num_classes)
    print("ĐÃ ĐƯỢC")
    print("load pretrain ")
    return model


def resnet101(num_classes=5, pretrained=True, vit_dim=768, vit_depth=6, vit_heads=8, vit_mlp=2048):
    resnet = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
    if pretrained:
        init_pretrained_weights(resnet, model_urls["resnet101"])

    vit = ViTEncoder(input_dim=resnet._out_features, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp)
    vit = load_vit_pretrain(vit)

    model = ResNetWithViT(resnet, vit, num_classes)
    return model


def resnet152(num_classes=5, pretrained=True, vit_dim=768, vit_depth=6, vit_heads=8, vit_mlp=2048):
    resnet = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])
    if pretrained:
        init_pretrained_weights(resnet, model_urls["resnet152"])

    vit = ViTEncoder(input_dim=resnet._out_features, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp)
    vit = load_vit_pretrain(vit)

    model = ResNetWithViT(resnet, vit, num_classes)
    return model

# def freeze_backbone(self):
#     for name, param in self.network.named_parameters():
#         if "resnet" in name or "vit" in name:
#             param.requires_grad = False

# def unfreeze_all(self):
#     for param in self.network.parameters():
#         param.requires_grad = True


# def build_optimizer(self, stage, cfg=None):
#     if stage == 1:
#         # Stage 1: chỉ train classifier head
#         params = [p for p in self.network.fc.parameters() if p.requires_grad]
#         self.optimizer = torch.optim.SGD(
#             params,
#             lr=cfg.LR_HEAD if cfg else 1e-2,
#             momentum=cfg.MOMENTUM if cfg else 0.9,
#             weight_decay=cfg.WEIGHT_DECAY if cfg else 1e-4,
#             nesterov=True
#         )

#     elif stage == 2:
#         # Stage 2: train toàn bộ backbone + head
#         resnet_params = [p for p in self.network.resnet.parameters() if p.requires_grad]
#         vit_params = [p for p in self.network.vit.parameters() if p.requires_grad]
#         fc_params = [p for p in self.network.fc.parameters() if p.requires_grad]

#         self.optimizer = torch.optim.SGD(
#             [
#                 {"params": resnet_params, "lr": cfg.LR_RESNET if cfg else 1e-4},
#                 {"params": vit_params, "lr": cfg.LR_VIT if cfg else 3e-4},
#                 {"params": fc_params, "lr": cfg.LR_HEAD if cfg else 1e-3}
#             ],
#             momentum=cfg.MOMENTUM if cfg else 0.9,
#             weight_decay=cfg.WEIGHT_DECAY if cfg else 1e-4,
#             nesterov=True
#         )


# def resnet18(pretrained=True, **kwargs):
#     model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

#     if pretrained:
#         init_pretrained_weights(model, model_urls["resnet18"])

#     return model


# def resnet34(pretrained=True, **kwargs):
#     model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

#     if pretrained:
#         init_pretrained_weights(model, model_urls["resnet34"])

#     return model


# def resnet50(pretrained=True, **kwargs):
#     model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

#     if pretrained:
#         init_pretrained_weights(model, model_urls["resnet50"])

#     return model


# def resnet101(pretrained=True, **kwargs):
#     model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

#     if pretrained:
#         init_pretrained_weights(model, model_urls["resnet101"])

#     return model


# def resnet152(pretrained=True, **kwargs):
#     model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

#     if pretrained:
#         init_pretrained_weights(model, model_urls["resnet152"])

#     return model