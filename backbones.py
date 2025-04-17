import timm  # noqa
import torch
import torchvision.models as models  # noqa



def load_ref_wrn50():
    
    import resnet 
    return resnet.wide_resnet50_2(True)

_BACKBONES = {
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "mc3_resnet50": "load_mc3_rn50()", 
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "ref_wideresnet50": "load_ref_wrn50()",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
}


def load(name):
    return eval(_BACKBONES[name])
