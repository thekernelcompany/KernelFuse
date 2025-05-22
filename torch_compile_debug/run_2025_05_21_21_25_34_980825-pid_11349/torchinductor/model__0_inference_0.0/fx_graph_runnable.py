
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_ubuntu'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0+cu126
# torch cuda version: 12.6
# torch git version: 134179474539648ba7dee1317959529fbd0e7f89


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Apr__9_19:24:57_PDT_2025 
# Cuda compilation tools, release 12.9, V12.9.41 
# Build cuda_12.9.r12.9/compiler.35813241_0 

# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1):
        iota = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, iota);  arg2_1 = iota = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view = torch.ops.aten.view.default(convert_element_type_1, [256, 768]);  convert_element_type_1 = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [4, 64, 2304]);  mm = None
        split = torch.ops.aten.split.Tensor(view_1, 768, 2);  view_1 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_2 = torch.ops.aten.view.default(getitem_3, [4, 64, 12, 64]);  getitem_3 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(getitem_2, [4, 64, 12, 64]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(getitem_4, [4, 64, 12, 64]);  getitem_4 = None
        permute_3 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_2, permute_1, permute_3, 0.0, True, scale = 0.125);  permute_2 = permute_1 = permute_3 = None
        getitem_5 = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
        permute_4 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_5 = torch.ops.aten.view.default(permute_4, [4, 64, 768]);  permute_4 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        permute_5 = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        view_6 = torch.ops.aten.view.default(view_5, [256, 768]);  view_5 = None
        mm_1 = torch.ops.aten.mm.default(view_6, permute_5);  view_6 = permute_5 = None
        view_7 = torch.ops.aten.view.default(mm_1, [4, 64, 768]);  mm_1 = None
        add_2 = torch.ops.aten.add.Tensor(add, view_7);  add = view_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_1[0]
        getitem_15 = var_mean_1[1];  var_mean_1 = None
        add_3 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_2, getitem_15);  getitem_15 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_6 = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        view_8 = torch.ops.aten.view.default(convert_element_type_8, [256, 768]);  convert_element_type_8 = None
        mm_2 = torch.ops.aten.mm.default(view_8, permute_6);  view_8 = permute_6 = None
        view_9 = torch.ops.aten.view.default(mm_2, [4, 64, 3072]);  mm_2 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476);  convert_element_type_11 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_7 = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        view_10 = torch.ops.aten.view.default(convert_element_type_12, [256, 3072]);  convert_element_type_12 = None
        mm_3 = torch.ops.aten.mm.default(view_10, permute_7);  view_10 = permute_7 = None
        view_11 = torch.ops.aten.view.default(mm_3, [4, 64, 768]);  mm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_2[0]
        getitem_17 = var_mean_2[1];  var_mean_2 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_5, getitem_17);  getitem_17 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg9_1);  mul_7 = arg9_1 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        permute_8 = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        view_12 = torch.ops.aten.view.default(convert_element_type_17, [256, 768]);  convert_element_type_17 = None
        mm_4 = torch.ops.aten.mm.default(view_12, permute_8);  view_12 = permute_8 = None
        view_13 = torch.ops.aten.view.default(mm_4, [4, 64, 2304]);  mm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_13, 768, 2);  view_13 = None
        getitem_18 = split_1[0]
        getitem_19 = split_1[1]
        getitem_20 = split_1[2];  split_1 = None
        view_14 = torch.ops.aten.view.default(getitem_19, [4, 64, 12, 64]);  getitem_19 = None
        permute_9 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        view_15 = torch.ops.aten.view.default(getitem_18, [4, 64, 12, 64]);  getitem_18 = None
        permute_10 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        view_16 = torch.ops.aten.view.default(getitem_20, [4, 64, 12, 64]);  getitem_20 = None
        permute_11 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_10, permute_9, permute_11, 0.0, True, scale = 0.125);  permute_10 = permute_9 = permute_11 = None
        getitem_21 = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
        permute_12 = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_17 = torch.ops.aten.view.default(permute_12, [4, 64, 768]);  permute_12 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        permute_13 = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        view_18 = torch.ops.aten.view.default(view_17, [256, 768]);  view_17 = None
        mm_5 = torch.ops.aten.mm.default(view_18, permute_13);  view_18 = permute_13 = None
        view_19 = torch.ops.aten.view.default(mm_5, [4, 64, 768]);  mm_5 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, view_19);  add_5 = view_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_3[0]
        getitem_31 = var_mean_3[1];  var_mean_3 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_31);  getitem_31 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg12_1);  mul_9 = arg12_1 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        permute_14 = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        view_20 = torch.ops.aten.view.default(convert_element_type_24, [256, 768]);  convert_element_type_24 = None
        mm_6 = torch.ops.aten.mm.default(view_20, permute_14);  view_20 = permute_14 = None
        view_21 = torch.ops.aten.view.default(mm_6, [4, 64, 3072]);  mm_6 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.7071067811865476);  convert_element_type_27 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = add_9 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_15 = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_28, [256, 3072]);  convert_element_type_28 = None
        mm_7 = torch.ops.aten.mm.default(view_22, permute_15);  view_22 = permute_15 = None
        view_23 = torch.ops.aten.view.default(mm_7, [4, 64, 768]);  mm_7 = None
        add_10 = torch.ops.aten.add.Tensor(add_7, view_23);  add_7 = view_23 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_4[0]
        getitem_33 = var_mean_4[1];  var_mean_4 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_10, getitem_33);  getitem_33 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg15_1);  mul_14 = arg15_1 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        permute_16 = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        view_24 = torch.ops.aten.view.default(convert_element_type_33, [256, 768]);  convert_element_type_33 = None
        mm_8 = torch.ops.aten.mm.default(view_24, permute_16);  view_24 = permute_16 = None
        view_25 = torch.ops.aten.view.default(mm_8, [4, 64, 2304]);  mm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_25, 768, 2);  view_25 = None
        getitem_34 = split_2[0]
        getitem_35 = split_2[1]
        getitem_36 = split_2[2];  split_2 = None
        view_26 = torch.ops.aten.view.default(getitem_35, [4, 64, 12, 64]);  getitem_35 = None
        permute_17 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        view_27 = torch.ops.aten.view.default(getitem_34, [4, 64, 12, 64]);  getitem_34 = None
        permute_18 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        view_28 = torch.ops.aten.view.default(getitem_36, [4, 64, 12, 64]);  getitem_36 = None
        permute_19 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_18, permute_17, permute_19, 0.0, True, scale = 0.125);  permute_18 = permute_17 = permute_19 = None
        getitem_37 = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
        permute_20 = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
        view_29 = torch.ops.aten.view.default(permute_20, [4, 64, 768]);  permute_20 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        permute_21 = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        view_30 = torch.ops.aten.view.default(view_29, [256, 768]);  view_29 = None
        mm_9 = torch.ops.aten.mm.default(view_30, permute_21);  view_30 = permute_21 = None
        view_31 = torch.ops.aten.view.default(mm_9, [4, 64, 768]);  mm_9 = None
        add_12 = torch.ops.aten.add.Tensor(add_10, view_31);  add_10 = view_31 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_5[0]
        getitem_47 = var_mean_5[1];  var_mean_5 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_12, getitem_47);  getitem_47 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg18_1);  mul_16 = arg18_1 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        permute_22 = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_40, [256, 768]);  convert_element_type_40 = None
        mm_10 = torch.ops.aten.mm.default(view_32, permute_22);  view_32 = permute_22 = None
        view_33 = torch.ops.aten.view.default(mm_10, [4, 64, 3072]);  mm_10 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        permute_23 = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_44, [256, 3072]);  convert_element_type_44 = None
        mm_11 = torch.ops.aten.mm.default(view_34, permute_23);  view_34 = permute_23 = None
        view_35 = torch.ops.aten.view.default(mm_11, [4, 64, 768]);  mm_11 = None
        add_15 = torch.ops.aten.add.Tensor(add_12, view_35);  add_12 = view_35 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_6[0]
        getitem_49 = var_mean_6[1];  var_mean_6 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_15, getitem_49);  getitem_49 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg21_1);  mul_21 = arg21_1 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        permute_24 = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_49, [256, 768]);  convert_element_type_49 = None
        mm_12 = torch.ops.aten.mm.default(view_36, permute_24);  view_36 = permute_24 = None
        view_37 = torch.ops.aten.view.default(mm_12, [4, 64, 2304]);  mm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_37, 768, 2);  view_37 = None
        getitem_50 = split_3[0]
        getitem_51 = split_3[1]
        getitem_52 = split_3[2];  split_3 = None
        view_38 = torch.ops.aten.view.default(getitem_51, [4, 64, 12, 64]);  getitem_51 = None
        permute_25 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        view_39 = torch.ops.aten.view.default(getitem_50, [4, 64, 12, 64]);  getitem_50 = None
        permute_26 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        view_40 = torch.ops.aten.view.default(getitem_52, [4, 64, 12, 64]);  getitem_52 = None
        permute_27 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_26, permute_25, permute_27, 0.0, True, scale = 0.125);  permute_26 = permute_25 = permute_27 = None
        getitem_53 = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
        permute_28 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_41 = torch.ops.aten.view.default(permute_28, [4, 64, 768]);  permute_28 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_29 = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        view_42 = torch.ops.aten.view.default(view_41, [256, 768]);  view_41 = None
        mm_13 = torch.ops.aten.mm.default(view_42, permute_29);  view_42 = permute_29 = None
        view_43 = torch.ops.aten.view.default(mm_13, [4, 64, 768]);  mm_13 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, view_43);  add_15 = view_43 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_7[0]
        getitem_63 = var_mean_7[1];  var_mean_7 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_17, getitem_63);  getitem_63 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, arg24_1);  mul_23 = arg24_1 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        permute_30 = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_56, [256, 768]);  convert_element_type_56 = None
        mm_14 = torch.ops.aten.mm.default(view_44, permute_30);  view_44 = permute_30 = None
        view_45 = torch.ops.aten.view.default(mm_14, [4, 64, 3072]);  mm_14 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476);  convert_element_type_59 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_19 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_19);  mul_25 = add_19 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        permute_31 = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        view_46 = torch.ops.aten.view.default(convert_element_type_60, [256, 3072]);  convert_element_type_60 = None
        mm_15 = torch.ops.aten.mm.default(view_46, permute_31);  view_46 = permute_31 = None
        view_47 = torch.ops.aten.view.default(mm_15, [4, 64, 768]);  mm_15 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, view_47);  add_17 = view_47 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_8[0]
        getitem_65 = var_mean_8[1];  var_mean_8 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_20, getitem_65);  getitem_65 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg27_1);  mul_28 = arg27_1 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        permute_32 = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_65, [256, 768]);  convert_element_type_65 = None
        mm_16 = torch.ops.aten.mm.default(view_48, permute_32);  view_48 = permute_32 = None
        view_49 = torch.ops.aten.view.default(mm_16, [4, 64, 2304]);  mm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_49, 768, 2);  view_49 = None
        getitem_66 = split_4[0]
        getitem_67 = split_4[1]
        getitem_68 = split_4[2];  split_4 = None
        view_50 = torch.ops.aten.view.default(getitem_67, [4, 64, 12, 64]);  getitem_67 = None
        permute_33 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(getitem_66, [4, 64, 12, 64]);  getitem_66 = None
        permute_34 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(getitem_68, [4, 64, 12, 64]);  getitem_68 = None
        permute_35 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_34, permute_33, permute_35, 0.0, True, scale = 0.125);  permute_34 = permute_33 = permute_35 = None
        getitem_69 = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
        permute_36 = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_53 = torch.ops.aten.view.default(permute_36, [4, 64, 768]);  permute_36 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        permute_37 = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_54 = torch.ops.aten.view.default(view_53, [256, 768]);  view_53 = None
        mm_17 = torch.ops.aten.mm.default(view_54, permute_37);  view_54 = permute_37 = None
        view_55 = torch.ops.aten.view.default(mm_17, [4, 64, 768]);  mm_17 = None
        add_22 = torch.ops.aten.add.Tensor(add_20, view_55);  add_20 = view_55 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_9[0]
        getitem_79 = var_mean_9[1];  var_mean_9 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_22, getitem_79);  getitem_79 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg30_1);  mul_30 = arg30_1 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        permute_38 = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        view_56 = torch.ops.aten.view.default(convert_element_type_72, [256, 768]);  convert_element_type_72 = None
        mm_18 = torch.ops.aten.mm.default(view_56, permute_38);  view_56 = permute_38 = None
        view_57 = torch.ops.aten.view.default(mm_18, [4, 64, 3072]);  mm_18 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476);  convert_element_type_75 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_24 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_24);  mul_32 = add_24 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        permute_39 = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        view_58 = torch.ops.aten.view.default(convert_element_type_76, [256, 3072]);  convert_element_type_76 = None
        mm_19 = torch.ops.aten.mm.default(view_58, permute_39);  view_58 = permute_39 = None
        view_59 = torch.ops.aten.view.default(mm_19, [4, 64, 768]);  mm_19 = None
        add_25 = torch.ops.aten.add.Tensor(add_22, view_59);  add_22 = view_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_10[0]
        getitem_81 = var_mean_10[1];  var_mean_10 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_81);  getitem_81 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg33_1);  mul_35 = arg33_1 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        permute_40 = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_60 = torch.ops.aten.view.default(convert_element_type_81, [256, 768]);  convert_element_type_81 = None
        mm_20 = torch.ops.aten.mm.default(view_60, permute_40);  view_60 = permute_40 = None
        view_61 = torch.ops.aten.view.default(mm_20, [4, 64, 2304]);  mm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_61, 768, 2);  view_61 = None
        getitem_82 = split_5[0]
        getitem_83 = split_5[1]
        getitem_84 = split_5[2];  split_5 = None
        view_62 = torch.ops.aten.view.default(getitem_83, [4, 64, 12, 64]);  getitem_83 = None
        permute_41 = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        view_63 = torch.ops.aten.view.default(getitem_82, [4, 64, 12, 64]);  getitem_82 = None
        permute_42 = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        view_64 = torch.ops.aten.view.default(getitem_84, [4, 64, 12, 64]);  getitem_84 = None
        permute_43 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_42, permute_41, permute_43, 0.0, True, scale = 0.125);  permute_42 = permute_41 = permute_43 = None
        getitem_85 = _scaled_dot_product_flash_attention_5[0];  _scaled_dot_product_flash_attention_5 = None
        permute_44 = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_65 = torch.ops.aten.view.default(permute_44, [4, 64, 768]);  permute_44 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        permute_45 = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        view_66 = torch.ops.aten.view.default(view_65, [256, 768]);  view_65 = None
        mm_21 = torch.ops.aten.mm.default(view_66, permute_45);  view_66 = permute_45 = None
        view_67 = torch.ops.aten.view.default(mm_21, [4, 64, 768]);  mm_21 = None
        add_27 = torch.ops.aten.add.Tensor(add_25, view_67);  add_25 = view_67 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_11[0]
        getitem_95 = var_mean_11[1];  var_mean_11 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_27, getitem_95);  getitem_95 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, arg36_1);  mul_37 = arg36_1 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        permute_46 = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_88, [256, 768]);  convert_element_type_88 = None
        mm_22 = torch.ops.aten.mm.default(view_68, permute_46);  view_68 = permute_46 = None
        view_69 = torch.ops.aten.view.default(mm_22, [4, 64, 3072]);  mm_22 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_29 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_29);  mul_39 = add_29 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        permute_47 = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        view_70 = torch.ops.aten.view.default(convert_element_type_92, [256, 3072]);  convert_element_type_92 = None
        mm_23 = torch.ops.aten.mm.default(view_70, permute_47);  view_70 = permute_47 = None
        view_71 = torch.ops.aten.view.default(mm_23, [4, 64, 768]);  mm_23 = None
        add_30 = torch.ops.aten.add.Tensor(add_27, view_71);  add_27 = view_71 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_12[0]
        getitem_97 = var_mean_12[1];  var_mean_12 = None
        add_31 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_30, getitem_97);  getitem_97 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg39_1);  mul_42 = arg39_1 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        permute_48 = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_72 = torch.ops.aten.view.default(convert_element_type_97, [256, 768]);  convert_element_type_97 = None
        mm_24 = torch.ops.aten.mm.default(view_72, permute_48);  view_72 = permute_48 = None
        view_73 = torch.ops.aten.view.default(mm_24, [4, 64, 2304]);  mm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_73, 768, 2);  view_73 = None
        getitem_98 = split_6[0]
        getitem_99 = split_6[1]
        getitem_100 = split_6[2];  split_6 = None
        view_74 = torch.ops.aten.view.default(getitem_99, [4, 64, 12, 64]);  getitem_99 = None
        permute_49 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        view_75 = torch.ops.aten.view.default(getitem_98, [4, 64, 12, 64]);  getitem_98 = None
        permute_50 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        view_76 = torch.ops.aten.view.default(getitem_100, [4, 64, 12, 64]);  getitem_100 = None
        permute_51 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_50, permute_49, permute_51, 0.0, True, scale = 0.125);  permute_50 = permute_49 = permute_51 = None
        getitem_101 = _scaled_dot_product_flash_attention_6[0];  _scaled_dot_product_flash_attention_6 = None
        permute_52 = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        view_77 = torch.ops.aten.view.default(permute_52, [4, 64, 768]);  permute_52 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        permute_53 = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        view_78 = torch.ops.aten.view.default(view_77, [256, 768]);  view_77 = None
        mm_25 = torch.ops.aten.mm.default(view_78, permute_53);  view_78 = permute_53 = None
        view_79 = torch.ops.aten.view.default(mm_25, [4, 64, 768]);  mm_25 = None
        add_32 = torch.ops.aten.add.Tensor(add_30, view_79);  add_30 = view_79 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_13[0]
        getitem_111 = var_mean_13[1];  var_mean_13 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_32, getitem_111);  getitem_111 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg42_1);  mul_44 = arg42_1 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        convert_element_type_104 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        permute_54 = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        view_80 = torch.ops.aten.view.default(convert_element_type_104, [256, 768]);  convert_element_type_104 = None
        mm_26 = torch.ops.aten.mm.default(view_80, permute_54);  view_80 = permute_54 = None
        view_81 = torch.ops.aten.view.default(mm_26, [4, 64, 3072]);  mm_26 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.7071067811865476);  convert_element_type_107 = None
        erf_6 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_34 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_34);  mul_46 = add_34 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        permute_55 = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        view_82 = torch.ops.aten.view.default(convert_element_type_108, [256, 3072]);  convert_element_type_108 = None
        mm_27 = torch.ops.aten.mm.default(view_82, permute_55);  view_82 = permute_55 = None
        view_83 = torch.ops.aten.view.default(mm_27, [4, 64, 768]);  mm_27 = None
        add_35 = torch.ops.aten.add.Tensor(add_32, view_83);  add_32 = view_83 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_112 = var_mean_14[0]
        getitem_113 = var_mean_14[1];  var_mean_14 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_35, getitem_113);  getitem_113 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg45_1);  mul_49 = arg45_1 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None
        permute_56 = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        view_84 = torch.ops.aten.view.default(convert_element_type_113, [256, 768]);  convert_element_type_113 = None
        mm_28 = torch.ops.aten.mm.default(view_84, permute_56);  view_84 = permute_56 = None
        view_85 = torch.ops.aten.view.default(mm_28, [4, 64, 2304]);  mm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_85, 768, 2);  view_85 = None
        getitem_114 = split_7[0]
        getitem_115 = split_7[1]
        getitem_116 = split_7[2];  split_7 = None
        view_86 = torch.ops.aten.view.default(getitem_115, [4, 64, 12, 64]);  getitem_115 = None
        permute_57 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        view_87 = torch.ops.aten.view.default(getitem_114, [4, 64, 12, 64]);  getitem_114 = None
        permute_58 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        view_88 = torch.ops.aten.view.default(getitem_116, [4, 64, 12, 64]);  getitem_116 = None
        permute_59 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_58, permute_57, permute_59, 0.0, True, scale = 0.125);  permute_58 = permute_57 = permute_59 = None
        getitem_117 = _scaled_dot_product_flash_attention_7[0];  _scaled_dot_product_flash_attention_7 = None
        permute_60 = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        view_89 = torch.ops.aten.view.default(permute_60, [4, 64, 768]);  permute_60 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        permute_61 = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        view_90 = torch.ops.aten.view.default(view_89, [256, 768]);  view_89 = None
        mm_29 = torch.ops.aten.mm.default(view_90, permute_61);  view_90 = permute_61 = None
        view_91 = torch.ops.aten.view.default(mm_29, [4, 64, 768]);  mm_29 = None
        add_37 = torch.ops.aten.add.Tensor(add_35, view_91);  add_35 = view_91 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_126 = var_mean_15[0]
        getitem_127 = var_mean_15[1];  var_mean_15 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_37, getitem_127);  getitem_127 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, arg48_1);  mul_51 = arg48_1 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        permute_62 = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_120, [256, 768]);  convert_element_type_120 = None
        mm_30 = torch.ops.aten.mm.default(view_92, permute_62);  view_92 = permute_62 = None
        view_93 = torch.ops.aten.view.default(mm_30, [4, 64, 3072]);  mm_30 = None
        convert_element_type_123 = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_53 = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476);  convert_element_type_123 = None
        erf_7 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_39 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_39);  mul_53 = add_39 = None
        convert_element_type_124 = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        permute_63 = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        view_94 = torch.ops.aten.view.default(convert_element_type_124, [256, 3072]);  convert_element_type_124 = None
        mm_31 = torch.ops.aten.mm.default(view_94, permute_63);  view_94 = permute_63 = None
        view_95 = torch.ops.aten.view.default(mm_31, [4, 64, 768]);  mm_31 = None
        add_40 = torch.ops.aten.add.Tensor(add_37, view_95);  add_37 = view_95 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_128 = var_mean_16[0]
        getitem_129 = var_mean_16[1];  var_mean_16 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_40, getitem_129);  getitem_129 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg51_1);  mul_56 = arg51_1 = None
        convert_element_type_128 = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_129 = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        permute_64 = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        view_96 = torch.ops.aten.view.default(convert_element_type_129, [256, 768]);  convert_element_type_129 = None
        mm_32 = torch.ops.aten.mm.default(view_96, permute_64);  view_96 = permute_64 = None
        view_97 = torch.ops.aten.view.default(mm_32, [4, 64, 2304]);  mm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_97, 768, 2);  view_97 = None
        getitem_130 = split_8[0]
        getitem_131 = split_8[1]
        getitem_132 = split_8[2];  split_8 = None
        view_98 = torch.ops.aten.view.default(getitem_131, [4, 64, 12, 64]);  getitem_131 = None
        permute_65 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        view_99 = torch.ops.aten.view.default(getitem_130, [4, 64, 12, 64]);  getitem_130 = None
        permute_66 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        view_100 = torch.ops.aten.view.default(getitem_132, [4, 64, 12, 64]);  getitem_132 = None
        permute_67 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_66, permute_65, permute_67, 0.0, True, scale = 0.125);  permute_66 = permute_65 = permute_67 = None
        getitem_133 = _scaled_dot_product_flash_attention_8[0];  _scaled_dot_product_flash_attention_8 = None
        permute_68 = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        view_101 = torch.ops.aten.view.default(permute_68, [4, 64, 768]);  permute_68 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        permute_69 = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        view_102 = torch.ops.aten.view.default(view_101, [256, 768]);  view_101 = None
        mm_33 = torch.ops.aten.mm.default(view_102, permute_69);  view_102 = permute_69 = None
        view_103 = torch.ops.aten.view.default(mm_33, [4, 64, 768]);  mm_33 = None
        add_42 = torch.ops.aten.add.Tensor(add_40, view_103);  add_40 = view_103 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_142 = var_mean_17[0]
        getitem_143 = var_mean_17[1];  var_mean_17 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_42, getitem_143);  getitem_143 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg54_1);  mul_58 = arg54_1 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        permute_70 = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_136, [256, 768]);  convert_element_type_136 = None
        mm_34 = torch.ops.aten.mm.default(view_104, permute_70);  view_104 = permute_70 = None
        view_105 = torch.ops.aten.view.default(mm_34, [4, 64, 3072]);  mm_34 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_61 = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_8 = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_44 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_60, add_44);  mul_60 = add_44 = None
        convert_element_type_140 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_141 = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        permute_71 = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_140, [256, 3072]);  convert_element_type_140 = None
        mm_35 = torch.ops.aten.mm.default(view_106, permute_71);  view_106 = permute_71 = None
        view_107 = torch.ops.aten.view.default(mm_35, [4, 64, 768]);  mm_35 = None
        add_45 = torch.ops.aten.add.Tensor(add_42, view_107);  add_42 = view_107 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_144 = var_mean_18[0]
        getitem_145 = var_mean_18[1];  var_mean_18 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_145);  getitem_145 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, arg57_1);  mul_63 = arg57_1 = None
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        permute_72 = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_145, [256, 768]);  convert_element_type_145 = None
        mm_36 = torch.ops.aten.mm.default(view_108, permute_72);  view_108 = permute_72 = None
        view_109 = torch.ops.aten.view.default(mm_36, [4, 64, 2304]);  mm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_109, 768, 2);  view_109 = None
        getitem_146 = split_9[0]
        getitem_147 = split_9[1]
        getitem_148 = split_9[2];  split_9 = None
        view_110 = torch.ops.aten.view.default(getitem_147, [4, 64, 12, 64]);  getitem_147 = None
        permute_73 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        view_111 = torch.ops.aten.view.default(getitem_146, [4, 64, 12, 64]);  getitem_146 = None
        permute_74 = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        view_112 = torch.ops.aten.view.default(getitem_148, [4, 64, 12, 64]);  getitem_148 = None
        permute_75 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_74, permute_73, permute_75, 0.0, True, scale = 0.125);  permute_74 = permute_73 = permute_75 = None
        getitem_149 = _scaled_dot_product_flash_attention_9[0];  _scaled_dot_product_flash_attention_9 = None
        permute_76 = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        view_113 = torch.ops.aten.view.default(permute_76, [4, 64, 768]);  permute_76 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        permute_77 = torch.ops.aten.permute.default(convert_element_type_148, [1, 0]);  convert_element_type_148 = None
        view_114 = torch.ops.aten.view.default(view_113, [256, 768]);  view_113 = None
        mm_37 = torch.ops.aten.mm.default(view_114, permute_77);  view_114 = permute_77 = None
        view_115 = torch.ops.aten.view.default(mm_37, [4, 64, 768]);  mm_37 = None
        add_47 = torch.ops.aten.add.Tensor(add_45, view_115);  add_45 = view_115 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_158 = var_mean_19[0]
        getitem_159 = var_mean_19[1];  var_mean_19 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_47, getitem_159);  getitem_159 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg60_1);  mul_65 = arg60_1 = None
        convert_element_type_151 = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        permute_78 = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        view_116 = torch.ops.aten.view.default(convert_element_type_152, [256, 768]);  convert_element_type_152 = None
        mm_38 = torch.ops.aten.mm.default(view_116, permute_78);  view_116 = permute_78 = None
        view_117 = torch.ops.aten.view.default(mm_38, [4, 64, 3072]);  mm_38 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476);  convert_element_type_155 = None
        erf_9 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_49 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_49);  mul_67 = add_49 = None
        convert_element_type_156 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        convert_element_type_157 = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_79 = torch.ops.aten.permute.default(convert_element_type_157, [1, 0]);  convert_element_type_157 = None
        view_118 = torch.ops.aten.view.default(convert_element_type_156, [256, 3072]);  convert_element_type_156 = None
        mm_39 = torch.ops.aten.mm.default(view_118, permute_79);  view_118 = permute_79 = None
        view_119 = torch.ops.aten.view.default(mm_39, [4, 64, 768]);  mm_39 = None
        add_50 = torch.ops.aten.add.Tensor(add_47, view_119);  add_47 = view_119 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_160 = var_mean_20[0]
        getitem_161 = var_mean_20[1];  var_mean_20 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_50, getitem_161);  getitem_161 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg63_1);  mul_70 = arg63_1 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        permute_80 = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_120 = torch.ops.aten.view.default(convert_element_type_161, [256, 768]);  convert_element_type_161 = None
        mm_40 = torch.ops.aten.mm.default(view_120, permute_80);  view_120 = permute_80 = None
        view_121 = torch.ops.aten.view.default(mm_40, [4, 64, 2304]);  mm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_121, 768, 2);  view_121 = None
        getitem_162 = split_10[0]
        getitem_163 = split_10[1]
        getitem_164 = split_10[2];  split_10 = None
        view_122 = torch.ops.aten.view.default(getitem_163, [4, 64, 12, 64]);  getitem_163 = None
        permute_81 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        view_123 = torch.ops.aten.view.default(getitem_162, [4, 64, 12, 64]);  getitem_162 = None
        permute_82 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        view_124 = torch.ops.aten.view.default(getitem_164, [4, 64, 12, 64]);  getitem_164 = None
        permute_83 = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_82, permute_81, permute_83, 0.0, True, scale = 0.125);  permute_82 = permute_81 = permute_83 = None
        getitem_165 = _scaled_dot_product_flash_attention_10[0];  _scaled_dot_product_flash_attention_10 = None
        permute_84 = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        view_125 = torch.ops.aten.view.default(permute_84, [4, 64, 768]);  permute_84 = None
        convert_element_type_164 = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        permute_85 = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        view_126 = torch.ops.aten.view.default(view_125, [256, 768]);  view_125 = None
        mm_41 = torch.ops.aten.mm.default(view_126, permute_85);  view_126 = permute_85 = None
        view_127 = torch.ops.aten.view.default(mm_41, [4, 64, 768]);  mm_41 = None
        add_52 = torch.ops.aten.add.Tensor(add_50, view_127);  add_50 = view_127 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_174 = var_mean_21[0]
        getitem_175 = var_mean_21[1];  var_mean_21 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_52, getitem_175);  getitem_175 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg66_1);  mul_72 = arg66_1 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_86 = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        view_128 = torch.ops.aten.view.default(convert_element_type_168, [256, 768]);  convert_element_type_168 = None
        mm_42 = torch.ops.aten.mm.default(view_128, permute_86);  view_128 = permute_86 = None
        view_129 = torch.ops.aten.view.default(mm_42, [4, 64, 3072]);  mm_42 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_74 = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.5)
        mul_75 = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.7071067811865476);  convert_element_type_171 = None
        erf_10 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_74, add_54);  mul_74 = add_54 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None
        convert_element_type_173 = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_87 = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        view_130 = torch.ops.aten.view.default(convert_element_type_172, [256, 3072]);  convert_element_type_172 = None
        mm_43 = torch.ops.aten.mm.default(view_130, permute_87);  view_130 = permute_87 = None
        view_131 = torch.ops.aten.view.default(mm_43, [4, 64, 768]);  mm_43 = None
        add_55 = torch.ops.aten.add.Tensor(add_52, view_131);  add_52 = view_131 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_176 = var_mean_22[0]
        getitem_177 = var_mean_22[1];  var_mean_22 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_55, getitem_177);  getitem_177 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, arg69_1);  mul_77 = arg69_1 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        permute_88 = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        view_132 = torch.ops.aten.view.default(convert_element_type_177, [256, 768]);  convert_element_type_177 = None
        mm_44 = torch.ops.aten.mm.default(view_132, permute_88);  view_132 = permute_88 = None
        view_133 = torch.ops.aten.view.default(mm_44, [4, 64, 2304]);  mm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_133, 768, 2);  view_133 = None
        getitem_178 = split_11[0]
        getitem_179 = split_11[1]
        getitem_180 = split_11[2];  split_11 = None
        view_134 = torch.ops.aten.view.default(getitem_179, [4, 64, 12, 64]);  getitem_179 = None
        permute_89 = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        view_135 = torch.ops.aten.view.default(getitem_178, [4, 64, 12, 64]);  getitem_178 = None
        permute_90 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        view_136 = torch.ops.aten.view.default(getitem_180, [4, 64, 12, 64]);  getitem_180 = None
        permute_91 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_90, permute_89, permute_91, 0.0, True, scale = 0.125);  permute_90 = permute_89 = permute_91 = None
        getitem_181 = _scaled_dot_product_flash_attention_11[0];  _scaled_dot_product_flash_attention_11 = None
        permute_92 = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_137 = torch.ops.aten.view.default(permute_92, [4, 64, 768]);  permute_92 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        permute_93 = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        view_138 = torch.ops.aten.view.default(view_137, [256, 768]);  view_137 = None
        mm_45 = torch.ops.aten.mm.default(view_138, permute_93);  view_138 = permute_93 = None
        view_139 = torch.ops.aten.view.default(mm_45, [4, 64, 768]);  mm_45 = None
        add_57 = torch.ops.aten.add.Tensor(add_55, view_139);  add_55 = view_139 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_190 = var_mean_23[0]
        getitem_191 = var_mean_23[1];  var_mean_23 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_57, getitem_191);  getitem_191 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, arg72_1);  mul_79 = arg72_1 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        permute_94 = torch.ops.aten.permute.default(convert_element_type_183, [1, 0]);  convert_element_type_183 = None
        view_140 = torch.ops.aten.view.default(convert_element_type_184, [256, 768]);  convert_element_type_184 = None
        mm_46 = torch.ops.aten.mm.default(view_140, permute_94);  view_140 = permute_94 = None
        view_141 = torch.ops.aten.view.default(mm_46, [4, 64, 3072]);  mm_46 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_81 = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5)
        mul_82 = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476);  convert_element_type_187 = None
        erf_11 = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_59 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_81, add_59);  mul_81 = add_59 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_95 = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        view_142 = torch.ops.aten.view.default(convert_element_type_188, [256, 3072]);  convert_element_type_188 = None
        mm_47 = torch.ops.aten.mm.default(view_142, permute_95);  view_142 = permute_95 = None
        view_143 = torch.ops.aten.view.default(mm_47, [4, 64, 768]);  mm_47 = None
        add_60 = torch.ops.aten.add.Tensor(add_57, view_143);  add_57 = view_143 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_192 = var_mean_24[0]
        getitem_193 = var_mean_24[1];  var_mean_24 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_60, getitem_193);  add_60 = getitem_193 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg75_1);  mul_84 = arg75_1 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        permute_96 = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        view_144 = torch.ops.aten.view.default(convert_element_type_193, [256, 768]);  convert_element_type_193 = None
        mm_48 = torch.ops.aten.mm.default(view_144, permute_96);  view_144 = permute_96 = None
        view_145 = torch.ops.aten.view.default(mm_48, [4, 64, 65]);  mm_48 = None
        view_146 = torch.ops.aten.view.default(view_145, [-1, 65])
        view_147 = torch.ops.aten.view.default(arg76_1, [-1]);  arg76_1 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        amax = torch.ops.aten.amax.default(convert_element_type_196, [1], True)
        sub_25 = torch.ops.aten.sub.Tensor(convert_element_type_196, amax);  convert_element_type_196 = amax = None
        exp = torch.ops.aten.exp.default(sub_25)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(sub_26, torch.float32);  sub_26 = None
        ne = torch.ops.aten.ne.Scalar(view_147, -1)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_147, full_default);  ne = full_default = None
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(convert_element_type_default, 1, unsqueeze);  convert_element_type_default = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_147, -1)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_147, -1);  view_147 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_199);  sum_3 = convert_element_type_199 = None
        return (view_145, div)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 64), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 199680, device=device(type='cuda', index=0))
    reader.tensor(buf1, (65, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2304, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf7, (3072, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 3072), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf10, (2304, 768), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (3072, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 3072), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2304, 768), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768, 768), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf19, (3072, 768), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 3072), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf22, (2304, 768), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 768), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf25, (3072, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 3072), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf28, (2304, 768), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768, 768), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf31, (3072, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 3072), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf34, (2304, 768), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf37, (3072, 768), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 3072), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2304, 768), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf43, (3072, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768, 3072), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2304, 768), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (3072, 768), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768, 3072), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf52, (2304, 768), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 768), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf55, (3072, 768), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 3072), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf58, (2304, 768), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768, 768), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf61, (3072, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768, 3072), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2304, 768), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768, 768), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf67, (3072, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 3072), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf70, (2304, 768), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf73, (3072, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 3072), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf76, (4, 64), dtype=torch.int64, is_leaf=True)  # arg76_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)