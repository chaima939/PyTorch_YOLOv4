import torch
import torch.nn.functional as f
def gp(fm1, fm2):
    h, w = fm2.shape[1], fm2.shape[2]
    global_ctx = torch.mean(fm1, axis=[1, 2], keepdim=True)
    global_ctx = nn.Sigmoid(global_ctx)
    m = nn.Upsample(size=[h,w]  , scale_factor=2, mode='bilinear',align_corners=False)
    out = (global_ctx * fm2) + m(fm1)
    return out 

def rcb(fm):
    fm = f.relu(fm)
    fm = nn.Conv2d(in_channels=fm, out_channels=384, kernel_size=3, stride=1, padding=1) # in_channels=output_filters[-1]
    fm = nn.BatchNorm2d(num_features=384,affine=True,momentum=0.03)
    return fm

def sum_fm(fm1, fm2):
    h, w = fm2.shape[1], fm2.shape[2]
    m = nn.Upsample(size=[h,w], scale_factor=2, mode='bilinear',align_corners=False)
    output = fm2 + m(fm1)
    return output
  

#2 is 0 // 3 is 1 // 4 is 2 // 5 is 3 // 4 is 6
def nas_fpn(feature_dict):
    GP_P5_P3 = gp(feature_dict[3], feature_dict[1])
    GP_P5_P3_RCB = rcb(GP_P5_P3)
    SUM1 = sum_fm(GP_P5_P3_RCB, feature_dict[1])
    SUM1_RCB = rcb(SUM1)
    SUM2 = sum_fm(SUM1_RCB, feature_dict[0])
    SUM2_RCB = rcb(SUM2)  # P2
    SUM3 = sum_fm(SUM2_RCB, SUM1_RCB)
    SUM3_RCB = rcb(SUM3)  # P3
    SUM3_RCB_GP = gp(SUM2_RCB, SUM3_RCB)
    SUM4 = sum_fm(SUM3_RCB_GP, feature_dict[2])
    SUM4_RCB = rcb(SUM4)  # P4
    SUM4_RCB_GP = gp(SUM1_RCB, SUM4_RCB)
    SUM5 = sum_fm(SUM4_RCB_GP, feature_dict[4])
    SUM5_RCB = rcb(SUM5)  # P6
    h, w = feature_dict[3].shape[1], feature_dict[3].shape[2]
    m = nn.Upsample(size=[h,w]  , scale_factor=2, mode='bilinear',align_corners=False)
    SUM5_RCB_resize = m(SUM5_RCB)
    SUM4_RCB_GP1 = gp(SUM4_RCB, SUM5_RCB_resize)
    SUM4_RCB_GP1_RCB = rcb(SUM4_RCB_GP1)  # P5
    pyramid_dict = {'P2': SUM2_RCB, 'P3': SUM3_RCB, 'P4': SUM4_RCB,
                    'P5': SUM4_RCB_GP1_RCB, 'P6': SUM5_RCB}
    return pyramid_dict
