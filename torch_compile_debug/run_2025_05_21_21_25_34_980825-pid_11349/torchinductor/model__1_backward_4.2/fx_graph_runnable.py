
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

torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



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

    
    
    def forward(self, primals_1, primals_4, primals_7, primals_10, primals_13, primals_16, primals_19, primals_22, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_77, iota, embedding, embedding_1, getitem_1, rsqrt, view, permute_1, permute_2, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mul_2, view_8, mm_2, view_10, mul_7, view_12, permute_9, permute_10, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_9, view_20, mm_6, view_22, mul_14, view_24, permute_17, permute_18, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_16, view_32, mm_10, view_34, mul_21, view_36, permute_25, permute_26, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_23, view_44, mm_14, view_46, mul_28, view_48, permute_33, permute_34, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_30, view_56, mm_18, view_58, mul_35, view_60, permute_41, permute_42, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_37, view_68, mm_22, view_70, mul_42, view_72, permute_49, permute_50, permute_51, getitem_101, getitem_102, getitem_107, getitem_108, mul_44, view_80, mm_26, view_82, mul_49, view_84, permute_57, permute_58, permute_59, getitem_117, getitem_118, getitem_123, getitem_124, mul_51, view_92, mm_30, view_94, mul_56, view_96, permute_65, permute_66, permute_67, getitem_133, getitem_134, getitem_139, getitem_140, mul_58, view_104, mm_34, view_106, mul_63, view_108, permute_73, permute_74, permute_75, getitem_149, getitem_150, getitem_155, getitem_156, mul_65, view_116, mm_38, view_118, mul_70, view_120, permute_81, permute_82, permute_83, getitem_165, getitem_166, getitem_171, getitem_172, mul_72, view_128, mm_42, view_130, mul_77, view_132, permute_89, permute_90, permute_91, getitem_181, getitem_182, getitem_187, getitem_188, mul_79, view_140, mm_46, view_142, mul_84, view_144, view_145, amax, log, convert_element_type_199, permute_99, div_2, permute_103, permute_107, div_3, permute_111, permute_119, div_4, permute_123, permute_127, div_5, permute_131, permute_139, div_6, permute_143, permute_147, div_7, permute_151, permute_159, div_8, permute_163, permute_167, div_9, permute_171, permute_179, div_10, permute_183, permute_187, div_11, permute_191, permute_199, div_12, permute_203, permute_207, div_13, permute_211, permute_219, div_14, permute_223, permute_227, div_15, permute_231, permute_239, div_16, permute_243, permute_247, div_17, permute_251, permute_259, div_18, permute_263, permute_267, div_19, permute_271, permute_279, div_20, permute_283, permute_287, div_21, permute_291, permute_299, div_22, permute_303, permute_307, div_23, permute_311, permute_319, div_24, permute_323, permute_327, div_25, permute_331, permute_339, tangents_1, tangents_2):
        div_1 = torch.ops.aten.div.Tensor(tangents_2, convert_element_type_199);  tangents_2 = convert_element_type_199 = None
        view_147 = torch.ops.aten.view.default(primals_77, [-1]);  primals_77 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_147, 1);  view_147 = None
        ne_3 = torch.ops.aten.ne.Scalar(unsqueeze_1, -1)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
        full_default_3 = torch.ops.aten.full.default([256, 65], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = None
        mul_86 = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_86, torch.float32);  mul_86 = None
        view_146 = torch.ops.aten.view.default(view_145, [-1, 65]);  view_145 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        sub_25 = torch.ops.aten.sub.Tensor(convert_element_type_196, amax);  convert_element_type_196 = amax = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_197 = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        convert_element_type_198 = torch.ops.prims.convert_element_type.default(convert_element_type_197, torch.float32);  convert_element_type_197 = None
        exp_1 = torch.ops.aten.exp.default(convert_element_type_198);  convert_element_type_198 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [1], True)
        mul_87 = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_27 = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_87);  convert_element_type_default = mul_87 = None
        convert_element_type_203 = torch.ops.prims.convert_element_type.default(sub_27, torch.bfloat16);  sub_27 = None
        view_148 = torch.ops.aten.view.default(convert_element_type_203, [4, 64, 65]);  convert_element_type_203 = None
        add_62 = torch.ops.aten.add.Tensor(tangents_1, view_148);  tangents_1 = view_148 = None
        view_149 = torch.ops.aten.view.default(add_62, [256, 65]);  add_62 = None
        permute_97 = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_49 = torch.ops.aten.mm.default(permute_97, view_144);  permute_97 = view_144 = None
        mm_50 = torch.ops.aten.mm.default(view_149, permute_99);  view_149 = permute_99 = None
        view_150 = torch.ops.aten.view.default(mm_50, [4, 64, 768]);  mm_50 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        convert_element_type_209 = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        mul_89 = torch.ops.aten.mul.Tensor(convert_element_type_208, primals_76);  primals_76 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, 768)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_89, [2], True)
        mul_91 = torch.ops.aten.mul.Tensor(mul_89, mul_84);  mul_89 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_91, [2], True);  mul_91 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_84, sum_6);  sum_6 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_90, sum_5);  mul_90 = sum_5 = None
        sub_30 = torch.ops.aten.sub.Tensor(sub_29, mul_92);  sub_29 = mul_92 = None
        mul_93 = torch.ops.aten.mul.Tensor(div_2, sub_30);  div_2 = sub_30 = None
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_208, mul_84);  convert_element_type_208 = mul_84 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1]);  mul_94 = None
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16)
        view_151 = torch.ops.aten.view.default(convert_element_type_210, [256, 768]);  convert_element_type_210 = None
        permute_101 = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_51 = torch.ops.aten.mm.default(permute_101, view_142);  permute_101 = view_142 = None
        mm_52 = torch.ops.aten.mm.default(view_151, permute_103);  view_151 = permute_103 = None
        view_152 = torch.ops.aten.view.default(mm_52, [4, 64, 3072]);  mm_52 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_216 = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        view_141 = torch.ops.aten.view.default(mm_46, [4, 64, 3072]);  mm_46 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_82 = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476)
        erf_11 = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_59 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96 = torch.ops.aten.mul.Tensor(add_59, 0.5);  add_59 = None
        mul_97 = torch.ops.aten.mul.Tensor(convert_element_type_187, convert_element_type_187)
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, -0.5);  mul_97 = None
        exp_2 = torch.ops.aten.exp.default(mul_98);  mul_98 = None
        mul_99 = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_187, mul_99);  convert_element_type_187 = mul_99 = None
        add_64 = torch.ops.aten.add.Tensor(mul_96, mul_100);  mul_96 = mul_100 = None
        mul_101 = torch.ops.aten.mul.Tensor(convert_element_type_216, add_64);  convert_element_type_216 = add_64 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        view_153 = torch.ops.aten.view.default(convert_element_type_218, [256, 3072]);  convert_element_type_218 = None
        permute_105 = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_53 = torch.ops.aten.mm.default(permute_105, view_140);  permute_105 = view_140 = None
        mm_54 = torch.ops.aten.mm.default(view_153, permute_107);  view_153 = permute_107 = None
        view_154 = torch.ops.aten.view.default(mm_54, [4, 64, 768]);  mm_54 = None
        convert_element_type_223 = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_223, primals_73);  primals_73 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_103, 768)
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_103, [2], True)
        mul_105 = torch.ops.aten.mul.Tensor(mul_103, mul_79);  mul_103 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_105, [2], True);  mul_105 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_79, sum_9);  sum_9 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_104, sum_8);  mul_104 = sum_8 = None
        sub_33 = torch.ops.aten.sub.Tensor(sub_32, mul_106);  sub_32 = mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(div_3, sub_33);  div_3 = sub_33 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_223, mul_79);  convert_element_type_223 = mul_79 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_108, [0, 1]);  mul_108 = None
        add_65 = torch.ops.aten.add.Tensor(mul_93, mul_107);  mul_93 = mul_107 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        view_155 = torch.ops.aten.view.default(convert_element_type_225, [256, 768]);  convert_element_type_225 = None
        permute_109 = torch.ops.aten.permute.default(view_155, [1, 0])
        permute_92 = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_137 = torch.ops.aten.view.default(permute_92, [4, 64, 768]);  permute_92 = None
        view_138 = torch.ops.aten.view.default(view_137, [256, 768]);  view_137 = None
        mm_55 = torch.ops.aten.mm.default(permute_109, view_138);  permute_109 = view_138 = None
        mm_56 = torch.ops.aten.mm.default(view_155, permute_111);  view_155 = permute_111 = None
        view_156 = torch.ops.aten.view.default(mm_56, [4, 64, 768]);  mm_56 = None
        convert_element_type_230 = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        view_157 = torch.ops.aten.view.default(view_156, [4, 64, 12, 64]);  view_156 = None
        permute_113 = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_113, permute_90, permute_89, permute_91, getitem_181, getitem_182, None, None, 64, 64, 0.0, True, getitem_187, getitem_188, scale = 0.125);  permute_113 = permute_90 = permute_89 = permute_91 = getitem_181 = getitem_182 = getitem_187 = getitem_188 = None
        getitem_194 = _scaled_dot_product_flash_attention_backward[0]
        getitem_195 = _scaled_dot_product_flash_attention_backward[1]
        getitem_196 = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        permute_114 = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None
        view_158 = torch.ops.aten.view.default(permute_114, [4, 64, 768]);  permute_114 = None
        permute_115 = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_159 = torch.ops.aten.view.default(permute_115, [4, 64, 768]);  permute_115 = None
        permute_116 = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
        view_160 = torch.ops.aten.view.default(permute_116, [4, 64, 768]);  permute_116 = None
        cat = torch.ops.aten.cat.default([view_159, view_160, view_158], 2);  view_159 = view_160 = view_158 = None
        view_161 = torch.ops.aten.view.default(cat, [256, 2304]);  cat = None
        permute_117 = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_57 = torch.ops.aten.mm.default(permute_117, view_132);  permute_117 = view_132 = None
        mm_58 = torch.ops.aten.mm.default(view_161, permute_119);  view_161 = permute_119 = None
        view_162 = torch.ops.aten.view.default(mm_58, [4, 64, 768]);  mm_58 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_235, primals_70);  primals_70 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, 768)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_110, [2], True)
        mul_112 = torch.ops.aten.mul.Tensor(mul_110, mul_77);  mul_110 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_112, [2], True);  mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_77, sum_12);  sum_12 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_111, sum_11);  mul_111 = sum_11 = None
        sub_36 = torch.ops.aten.sub.Tensor(sub_35, mul_113);  sub_35 = mul_113 = None
        mul_114 = torch.ops.aten.mul.Tensor(div_4, sub_36);  div_4 = sub_36 = None
        mul_115 = torch.ops.aten.mul.Tensor(convert_element_type_235, mul_77);  convert_element_type_235 = mul_77 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1]);  mul_115 = None
        add_66 = torch.ops.aten.add.Tensor(add_65, mul_114);  add_65 = mul_114 = None
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16)
        view_163 = torch.ops.aten.view.default(convert_element_type_237, [256, 768]);  convert_element_type_237 = None
        permute_121 = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_59 = torch.ops.aten.mm.default(permute_121, view_130);  permute_121 = view_130 = None
        mm_60 = torch.ops.aten.mm.default(view_163, permute_123);  view_163 = permute_123 = None
        view_164 = torch.ops.aten.view.default(mm_60, [4, 64, 3072]);  mm_60 = None
        convert_element_type_242 = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_243 = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        view_129 = torch.ops.aten.view.default(mm_42, [4, 64, 3072]);  mm_42 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_75 = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.7071067811865476)
        erf_10 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_117 = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_171, convert_element_type_171)
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, -0.5);  mul_118 = None
        exp_3 = torch.ops.aten.exp.default(mul_119);  mul_119 = None
        mul_120 = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_121 = torch.ops.aten.mul.Tensor(convert_element_type_171, mul_120);  convert_element_type_171 = mul_120 = None
        add_68 = torch.ops.aten.add.Tensor(mul_117, mul_121);  mul_117 = mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(convert_element_type_243, add_68);  convert_element_type_243 = add_68 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        view_165 = torch.ops.aten.view.default(convert_element_type_245, [256, 3072]);  convert_element_type_245 = None
        permute_125 = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_61 = torch.ops.aten.mm.default(permute_125, view_128);  permute_125 = view_128 = None
        mm_62 = torch.ops.aten.mm.default(view_165, permute_127);  view_165 = permute_127 = None
        view_166 = torch.ops.aten.view.default(mm_62, [4, 64, 768]);  mm_62 = None
        convert_element_type_250 = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_250, primals_67);  primals_67 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, 768)
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_124, [2], True)
        mul_126 = torch.ops.aten.mul.Tensor(mul_124, mul_72);  mul_124 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_126, [2], True);  mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_72, sum_15);  sum_15 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_125, sum_14);  mul_125 = sum_14 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, mul_127);  sub_38 = mul_127 = None
        mul_128 = torch.ops.aten.mul.Tensor(div_5, sub_39);  div_5 = sub_39 = None
        mul_129 = torch.ops.aten.mul.Tensor(convert_element_type_250, mul_72);  convert_element_type_250 = mul_72 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_129, [0, 1]);  mul_129 = None
        add_69 = torch.ops.aten.add.Tensor(add_66, mul_128);  add_66 = mul_128 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        view_167 = torch.ops.aten.view.default(convert_element_type_252, [256, 768]);  convert_element_type_252 = None
        permute_129 = torch.ops.aten.permute.default(view_167, [1, 0])
        permute_84 = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_125 = torch.ops.aten.view.default(permute_84, [4, 64, 768]);  permute_84 = None
        view_126 = torch.ops.aten.view.default(view_125, [256, 768]);  view_125 = None
        mm_63 = torch.ops.aten.mm.default(permute_129, view_126);  permute_129 = view_126 = None
        mm_64 = torch.ops.aten.mm.default(view_167, permute_131);  view_167 = permute_131 = None
        view_168 = torch.ops.aten.view.default(mm_64, [4, 64, 768]);  mm_64 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        view_169 = torch.ops.aten.view.default(view_168, [4, 64, 12, 64]);  view_168 = None
        permute_133 = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_133, permute_82, permute_81, permute_83, getitem_165, getitem_166, None, None, 64, 64, 0.0, True, getitem_171, getitem_172, scale = 0.125);  permute_133 = permute_82 = permute_81 = permute_83 = getitem_165 = getitem_166 = getitem_171 = getitem_172 = None
        getitem_197 = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_198 = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_199 = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        permute_134 = torch.ops.aten.permute.default(getitem_199, [0, 2, 1, 3]);  getitem_199 = None
        view_170 = torch.ops.aten.view.default(permute_134, [4, 64, 768]);  permute_134 = None
        permute_135 = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        view_171 = torch.ops.aten.view.default(permute_135, [4, 64, 768]);  permute_135 = None
        permute_136 = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None
        view_172 = torch.ops.aten.view.default(permute_136, [4, 64, 768]);  permute_136 = None
        cat_1 = torch.ops.aten.cat.default([view_171, view_172, view_170], 2);  view_171 = view_172 = view_170 = None
        view_173 = torch.ops.aten.view.default(cat_1, [256, 2304]);  cat_1 = None
        permute_137 = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_65 = torch.ops.aten.mm.default(permute_137, view_120);  permute_137 = view_120 = None
        mm_66 = torch.ops.aten.mm.default(view_173, permute_139);  view_173 = permute_139 = None
        view_174 = torch.ops.aten.view.default(mm_66, [4, 64, 768]);  mm_66 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        convert_element_type_263 = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        mul_131 = torch.ops.aten.mul.Tensor(convert_element_type_262, primals_64);  primals_64 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, 768)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_131, [2], True)
        mul_133 = torch.ops.aten.mul.Tensor(mul_131, mul_70);  mul_131 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_133, [2], True);  mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_70, sum_18);  sum_18 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_132, sum_17);  mul_132 = sum_17 = None
        sub_42 = torch.ops.aten.sub.Tensor(sub_41, mul_134);  sub_41 = mul_134 = None
        mul_135 = torch.ops.aten.mul.Tensor(div_6, sub_42);  div_6 = sub_42 = None
        mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_262, mul_70);  convert_element_type_262 = mul_70 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_136, [0, 1]);  mul_136 = None
        add_70 = torch.ops.aten.add.Tensor(add_69, mul_135);  add_69 = mul_135 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16)
        view_175 = torch.ops.aten.view.default(convert_element_type_264, [256, 768]);  convert_element_type_264 = None
        permute_141 = torch.ops.aten.permute.default(view_175, [1, 0])
        mm_67 = torch.ops.aten.mm.default(permute_141, view_118);  permute_141 = view_118 = None
        mm_68 = torch.ops.aten.mm.default(view_175, permute_143);  view_175 = permute_143 = None
        view_176 = torch.ops.aten.view.default(mm_68, [4, 64, 3072]);  mm_68 = None
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(view_176, torch.float32);  view_176 = None
        view_117 = torch.ops.aten.view.default(mm_38, [4, 64, 3072]);  mm_38 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476)
        erf_9 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_49 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_138 = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_155)
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, -0.5);  mul_139 = None
        exp_4 = torch.ops.aten.exp.default(mul_140);  mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_155, mul_141);  convert_element_type_155 = mul_141 = None
        add_72 = torch.ops.aten.add.Tensor(mul_138, mul_142);  mul_138 = mul_142 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_270, add_72);  convert_element_type_270 = add_72 = None
        convert_element_type_272 = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        view_177 = torch.ops.aten.view.default(convert_element_type_272, [256, 3072]);  convert_element_type_272 = None
        permute_145 = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_69 = torch.ops.aten.mm.default(permute_145, view_116);  permute_145 = view_116 = None
        mm_70 = torch.ops.aten.mm.default(view_177, permute_147);  view_177 = permute_147 = None
        view_178 = torch.ops.aten.view.default(mm_70, [4, 64, 768]);  mm_70 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        mul_145 = torch.ops.aten.mul.Tensor(convert_element_type_277, primals_61);  primals_61 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, 768)
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147 = torch.ops.aten.mul.Tensor(mul_145, mul_65);  mul_145 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_65, sum_21);  sum_21 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_146, sum_20);  mul_146 = sum_20 = None
        sub_45 = torch.ops.aten.sub.Tensor(sub_44, mul_148);  sub_44 = mul_148 = None
        mul_149 = torch.ops.aten.mul.Tensor(div_7, sub_45);  div_7 = sub_45 = None
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_277, mul_65);  convert_element_type_277 = mul_65 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        add_73 = torch.ops.aten.add.Tensor(add_70, mul_149);  add_70 = mul_149 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16)
        view_179 = torch.ops.aten.view.default(convert_element_type_279, [256, 768]);  convert_element_type_279 = None
        permute_149 = torch.ops.aten.permute.default(view_179, [1, 0])
        permute_76 = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_113 = torch.ops.aten.view.default(permute_76, [4, 64, 768]);  permute_76 = None
        view_114 = torch.ops.aten.view.default(view_113, [256, 768]);  view_113 = None
        mm_71 = torch.ops.aten.mm.default(permute_149, view_114);  permute_149 = view_114 = None
        mm_72 = torch.ops.aten.mm.default(view_179, permute_151);  view_179 = permute_151 = None
        view_180 = torch.ops.aten.view.default(mm_72, [4, 64, 768]);  mm_72 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        view_181 = torch.ops.aten.view.default(view_180, [4, 64, 12, 64]);  view_180 = None
        permute_153 = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_153, permute_74, permute_73, permute_75, getitem_149, getitem_150, None, None, 64, 64, 0.0, True, getitem_155, getitem_156, scale = 0.125);  permute_153 = permute_74 = permute_73 = permute_75 = getitem_149 = getitem_150 = getitem_155 = getitem_156 = None
        getitem_200 = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_201 = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_202 = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        permute_154 = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
        view_182 = torch.ops.aten.view.default(permute_154, [4, 64, 768]);  permute_154 = None
        permute_155 = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
        view_183 = torch.ops.aten.view.default(permute_155, [4, 64, 768]);  permute_155 = None
        permute_156 = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
        view_184 = torch.ops.aten.view.default(permute_156, [4, 64, 768]);  permute_156 = None
        cat_2 = torch.ops.aten.cat.default([view_183, view_184, view_182], 2);  view_183 = view_184 = view_182 = None
        view_185 = torch.ops.aten.view.default(cat_2, [256, 2304]);  cat_2 = None
        permute_157 = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_73 = torch.ops.aten.mm.default(permute_157, view_108);  permute_157 = view_108 = None
        mm_74 = torch.ops.aten.mm.default(view_185, permute_159);  view_185 = permute_159 = None
        view_186 = torch.ops.aten.view.default(mm_74, [4, 64, 768]);  mm_74 = None
        convert_element_type_289 = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_289, primals_58);  primals_58 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154 = torch.ops.aten.mul.Tensor(mul_152, mul_63);  mul_152 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_63, sum_24);  sum_24 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_153, sum_23);  mul_153 = sum_23 = None
        sub_48 = torch.ops.aten.sub.Tensor(sub_47, mul_155);  sub_47 = mul_155 = None
        mul_156 = torch.ops.aten.mul.Tensor(div_8, sub_48);  div_8 = sub_48 = None
        mul_157 = torch.ops.aten.mul.Tensor(convert_element_type_289, mul_63);  convert_element_type_289 = mul_63 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        add_74 = torch.ops.aten.add.Tensor(add_73, mul_156);  add_73 = mul_156 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16)
        view_187 = torch.ops.aten.view.default(convert_element_type_291, [256, 768]);  convert_element_type_291 = None
        permute_161 = torch.ops.aten.permute.default(view_187, [1, 0])
        mm_75 = torch.ops.aten.mm.default(permute_161, view_106);  permute_161 = view_106 = None
        mm_76 = torch.ops.aten.mm.default(view_187, permute_163);  view_187 = permute_163 = None
        view_188 = torch.ops.aten.view.default(mm_76, [4, 64, 3072]);  mm_76 = None
        convert_element_type_296 = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_297 = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        view_105 = torch.ops.aten.view.default(mm_34, [4, 64, 3072]);  mm_34 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_61 = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476)
        erf_8 = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_44 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_159 = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_139, convert_element_type_139)
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
        exp_5 = torch.ops.aten.exp.default(mul_161);  mul_161 = None
        mul_162 = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_163 = torch.ops.aten.mul.Tensor(convert_element_type_139, mul_162);  convert_element_type_139 = mul_162 = None
        add_76 = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
        mul_164 = torch.ops.aten.mul.Tensor(convert_element_type_297, add_76);  convert_element_type_297 = add_76 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None
        view_189 = torch.ops.aten.view.default(convert_element_type_299, [256, 3072]);  convert_element_type_299 = None
        permute_165 = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_77 = torch.ops.aten.mm.default(permute_165, view_104);  permute_165 = view_104 = None
        mm_78 = torch.ops.aten.mm.default(view_189, permute_167);  view_189 = permute_167 = None
        view_190 = torch.ops.aten.view.default(mm_78, [4, 64, 768]);  mm_78 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        convert_element_type_305 = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_304, primals_55);  primals_55 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, 768)
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
        mul_168 = torch.ops.aten.mul.Tensor(mul_166, mul_58);  mul_166 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_58, sum_27);  sum_27 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_167, sum_26);  mul_167 = sum_26 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, mul_169);  sub_50 = mul_169 = None
        mul_170 = torch.ops.aten.mul.Tensor(div_9, sub_51);  div_9 = sub_51 = None
        mul_171 = torch.ops.aten.mul.Tensor(convert_element_type_304, mul_58);  convert_element_type_304 = mul_58 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
        add_77 = torch.ops.aten.add.Tensor(add_74, mul_170);  add_74 = mul_170 = None
        convert_element_type_306 = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        view_191 = torch.ops.aten.view.default(convert_element_type_306, [256, 768]);  convert_element_type_306 = None
        permute_169 = torch.ops.aten.permute.default(view_191, [1, 0])
        permute_68 = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_101 = torch.ops.aten.view.default(permute_68, [4, 64, 768]);  permute_68 = None
        view_102 = torch.ops.aten.view.default(view_101, [256, 768]);  view_101 = None
        mm_79 = torch.ops.aten.mm.default(permute_169, view_102);  permute_169 = view_102 = None
        mm_80 = torch.ops.aten.mm.default(view_191, permute_171);  view_191 = permute_171 = None
        view_192 = torch.ops.aten.view.default(mm_80, [4, 64, 768]);  mm_80 = None
        convert_element_type_311 = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        view_193 = torch.ops.aten.view.default(view_192, [4, 64, 12, 64]);  view_192 = None
        permute_173 = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_173, permute_66, permute_65, permute_67, getitem_133, getitem_134, None, None, 64, 64, 0.0, True, getitem_139, getitem_140, scale = 0.125);  permute_173 = permute_66 = permute_65 = permute_67 = getitem_133 = getitem_134 = getitem_139 = getitem_140 = None
        getitem_203 = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_204 = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_205 = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        permute_174 = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_194 = torch.ops.aten.view.default(permute_174, [4, 64, 768]);  permute_174 = None
        permute_175 = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None
        view_195 = torch.ops.aten.view.default(permute_175, [4, 64, 768]);  permute_175 = None
        permute_176 = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
        view_196 = torch.ops.aten.view.default(permute_176, [4, 64, 768]);  permute_176 = None
        cat_3 = torch.ops.aten.cat.default([view_195, view_196, view_194], 2);  view_195 = view_196 = view_194 = None
        view_197 = torch.ops.aten.view.default(cat_3, [256, 2304]);  cat_3 = None
        permute_177 = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_81 = torch.ops.aten.mm.default(permute_177, view_96);  permute_177 = view_96 = None
        mm_82 = torch.ops.aten.mm.default(view_197, permute_179);  view_197 = permute_179 = None
        view_198 = torch.ops.aten.view.default(mm_82, [4, 64, 768]);  mm_82 = None
        convert_element_type_316 = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        mul_173 = torch.ops.aten.mul.Tensor(convert_element_type_316, primals_52);  primals_52 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, 768)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
        mul_175 = torch.ops.aten.mul.Tensor(mul_173, mul_56);  mul_173 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_56, sum_30);  sum_30 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_174, sum_29);  mul_174 = sum_29 = None
        sub_54 = torch.ops.aten.sub.Tensor(sub_53, mul_176);  sub_53 = mul_176 = None
        mul_177 = torch.ops.aten.mul.Tensor(div_10, sub_54);  div_10 = sub_54 = None
        mul_178 = torch.ops.aten.mul.Tensor(convert_element_type_316, mul_56);  convert_element_type_316 = mul_56 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
        add_78 = torch.ops.aten.add.Tensor(add_77, mul_177);  add_77 = mul_177 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        view_199 = torch.ops.aten.view.default(convert_element_type_318, [256, 768]);  convert_element_type_318 = None
        permute_181 = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_83 = torch.ops.aten.mm.default(permute_181, view_94);  permute_181 = view_94 = None
        mm_84 = torch.ops.aten.mm.default(view_199, permute_183);  view_199 = permute_183 = None
        view_200 = torch.ops.aten.view.default(mm_84, [4, 64, 3072]);  mm_84 = None
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(view_200, torch.float32);  view_200 = None
        view_93 = torch.ops.aten.view.default(mm_30, [4, 64, 3072]);  mm_30 = None
        convert_element_type_123 = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476)
        erf_7 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_39 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_180 = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_181 = torch.ops.aten.mul.Tensor(convert_element_type_123, convert_element_type_123)
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, -0.5);  mul_181 = None
        exp_6 = torch.ops.aten.exp.default(mul_182);  mul_182 = None
        mul_183 = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_184 = torch.ops.aten.mul.Tensor(convert_element_type_123, mul_183);  convert_element_type_123 = mul_183 = None
        add_80 = torch.ops.aten.add.Tensor(mul_180, mul_184);  mul_180 = mul_184 = None
        mul_185 = torch.ops.aten.mul.Tensor(convert_element_type_324, add_80);  convert_element_type_324 = add_80 = None
        convert_element_type_326 = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        view_201 = torch.ops.aten.view.default(convert_element_type_326, [256, 3072]);  convert_element_type_326 = None
        permute_185 = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_85 = torch.ops.aten.mm.default(permute_185, view_92);  permute_185 = view_92 = None
        mm_86 = torch.ops.aten.mm.default(view_201, permute_187);  view_201 = permute_187 = None
        view_202 = torch.ops.aten.view.default(mm_86, [4, 64, 768]);  mm_86 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(view_202, torch.float32);  view_202 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        mul_187 = torch.ops.aten.mul.Tensor(convert_element_type_331, primals_49);  primals_49 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, 768)
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_187, [2], True)
        mul_189 = torch.ops.aten.mul.Tensor(mul_187, mul_51);  mul_187 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_189, [2], True);  mul_189 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_51, sum_33);  sum_33 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_188, sum_32);  mul_188 = sum_32 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, mul_190);  sub_56 = mul_190 = None
        mul_191 = torch.ops.aten.mul.Tensor(div_11, sub_57);  div_11 = sub_57 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_331, mul_51);  convert_element_type_331 = mul_51 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1]);  mul_192 = None
        add_81 = torch.ops.aten.add.Tensor(add_78, mul_191);  add_78 = mul_191 = None
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16)
        view_203 = torch.ops.aten.view.default(convert_element_type_333, [256, 768]);  convert_element_type_333 = None
        permute_189 = torch.ops.aten.permute.default(view_203, [1, 0])
        permute_60 = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_89 = torch.ops.aten.view.default(permute_60, [4, 64, 768]);  permute_60 = None
        view_90 = torch.ops.aten.view.default(view_89, [256, 768]);  view_89 = None
        mm_87 = torch.ops.aten.mm.default(permute_189, view_90);  permute_189 = view_90 = None
        mm_88 = torch.ops.aten.mm.default(view_203, permute_191);  view_203 = permute_191 = None
        view_204 = torch.ops.aten.view.default(mm_88, [4, 64, 768]);  mm_88 = None
        convert_element_type_338 = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        view_205 = torch.ops.aten.view.default(view_204, [4, 64, 12, 64]);  view_204 = None
        permute_193 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_193, permute_58, permute_57, permute_59, getitem_117, getitem_118, None, None, 64, 64, 0.0, True, getitem_123, getitem_124, scale = 0.125);  permute_193 = permute_58 = permute_57 = permute_59 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_206 = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_207 = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_208 = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        permute_194 = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
        view_206 = torch.ops.aten.view.default(permute_194, [4, 64, 768]);  permute_194 = None
        permute_195 = torch.ops.aten.permute.default(getitem_206, [0, 2, 1, 3]);  getitem_206 = None
        view_207 = torch.ops.aten.view.default(permute_195, [4, 64, 768]);  permute_195 = None
        permute_196 = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
        view_208 = torch.ops.aten.view.default(permute_196, [4, 64, 768]);  permute_196 = None
        cat_4 = torch.ops.aten.cat.default([view_207, view_208, view_206], 2);  view_207 = view_208 = view_206 = None
        view_209 = torch.ops.aten.view.default(cat_4, [256, 2304]);  cat_4 = None
        permute_197 = torch.ops.aten.permute.default(view_209, [1, 0])
        mm_89 = torch.ops.aten.mm.default(permute_197, view_84);  permute_197 = view_84 = None
        mm_90 = torch.ops.aten.mm.default(view_209, permute_199);  view_209 = permute_199 = None
        view_210 = torch.ops.aten.view.default(mm_90, [4, 64, 768]);  mm_90 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(view_210, torch.float32);  view_210 = None
        convert_element_type_344 = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        mul_194 = torch.ops.aten.mul.Tensor(convert_element_type_343, primals_46);  primals_46 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, 768)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_194, [2], True)
        mul_196 = torch.ops.aten.mul.Tensor(mul_194, mul_49);  mul_194 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_196, [2], True);  mul_196 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_49, sum_36);  sum_36 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_195, sum_35);  mul_195 = sum_35 = None
        sub_60 = torch.ops.aten.sub.Tensor(sub_59, mul_197);  sub_59 = mul_197 = None
        mul_198 = torch.ops.aten.mul.Tensor(div_12, sub_60);  div_12 = sub_60 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_343, mul_49);  convert_element_type_343 = mul_49 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
        add_82 = torch.ops.aten.add.Tensor(add_81, mul_198);  add_81 = mul_198 = None
        convert_element_type_345 = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16)
        view_211 = torch.ops.aten.view.default(convert_element_type_345, [256, 768]);  convert_element_type_345 = None
        permute_201 = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_91 = torch.ops.aten.mm.default(permute_201, view_82);  permute_201 = view_82 = None
        mm_92 = torch.ops.aten.mm.default(view_211, permute_203);  view_211 = permute_203 = None
        view_212 = torch.ops.aten.view.default(mm_92, [4, 64, 3072]);  mm_92 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        view_81 = torch.ops.aten.view.default(mm_26, [4, 64, 3072]);  mm_26 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.7071067811865476)
        erf_6 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_34 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_201 = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_202 = torch.ops.aten.mul.Tensor(convert_element_type_107, convert_element_type_107)
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, -0.5);  mul_202 = None
        exp_7 = torch.ops.aten.exp.default(mul_203);  mul_203 = None
        mul_204 = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_205 = torch.ops.aten.mul.Tensor(convert_element_type_107, mul_204);  convert_element_type_107 = mul_204 = None
        add_84 = torch.ops.aten.add.Tensor(mul_201, mul_205);  mul_201 = mul_205 = None
        mul_206 = torch.ops.aten.mul.Tensor(convert_element_type_351, add_84);  convert_element_type_351 = add_84 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        view_213 = torch.ops.aten.view.default(convert_element_type_353, [256, 3072]);  convert_element_type_353 = None
        permute_205 = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_93 = torch.ops.aten.mm.default(permute_205, view_80);  permute_205 = view_80 = None
        mm_94 = torch.ops.aten.mm.default(view_213, permute_207);  view_213 = permute_207 = None
        view_214 = torch.ops.aten.view.default(mm_94, [4, 64, 768]);  mm_94 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        convert_element_type_359 = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_358, primals_43);  primals_43 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, 768)
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
        mul_210 = torch.ops.aten.mul.Tensor(mul_208, mul_44);  mul_208 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_44, sum_39);  sum_39 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_209, sum_38);  mul_209 = sum_38 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, mul_211);  sub_62 = mul_211 = None
        mul_212 = torch.ops.aten.mul.Tensor(div_13, sub_63);  div_13 = sub_63 = None
        mul_213 = torch.ops.aten.mul.Tensor(convert_element_type_358, mul_44);  convert_element_type_358 = mul_44 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
        add_85 = torch.ops.aten.add.Tensor(add_82, mul_212);  add_82 = mul_212 = None
        convert_element_type_360 = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_215 = torch.ops.aten.view.default(convert_element_type_360, [256, 768]);  convert_element_type_360 = None
        permute_209 = torch.ops.aten.permute.default(view_215, [1, 0])
        permute_52 = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_77 = torch.ops.aten.view.default(permute_52, [4, 64, 768]);  permute_52 = None
        view_78 = torch.ops.aten.view.default(view_77, [256, 768]);  view_77 = None
        mm_95 = torch.ops.aten.mm.default(permute_209, view_78);  permute_209 = view_78 = None
        mm_96 = torch.ops.aten.mm.default(view_215, permute_211);  view_215 = permute_211 = None
        view_216 = torch.ops.aten.view.default(mm_96, [4, 64, 768]);  mm_96 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        view_217 = torch.ops.aten.view.default(view_216, [4, 64, 12, 64]);  view_216 = None
        permute_213 = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_213, permute_50, permute_49, permute_51, getitem_101, getitem_102, None, None, 64, 64, 0.0, True, getitem_107, getitem_108, scale = 0.125);  permute_213 = permute_50 = permute_49 = permute_51 = getitem_101 = getitem_102 = getitem_107 = getitem_108 = None
        getitem_209 = _scaled_dot_product_flash_attention_backward_5[0]
        getitem_210 = _scaled_dot_product_flash_attention_backward_5[1]
        getitem_211 = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
        permute_214 = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
        view_218 = torch.ops.aten.view.default(permute_214, [4, 64, 768]);  permute_214 = None
        permute_215 = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
        view_219 = torch.ops.aten.view.default(permute_215, [4, 64, 768]);  permute_215 = None
        permute_216 = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None
        view_220 = torch.ops.aten.view.default(permute_216, [4, 64, 768]);  permute_216 = None
        cat_5 = torch.ops.aten.cat.default([view_219, view_220, view_218], 2);  view_219 = view_220 = view_218 = None
        view_221 = torch.ops.aten.view.default(cat_5, [256, 2304]);  cat_5 = None
        permute_217 = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_97 = torch.ops.aten.mm.default(permute_217, view_72);  permute_217 = view_72 = None
        mm_98 = torch.ops.aten.mm.default(view_221, permute_219);  view_221 = permute_219 = None
        view_222 = torch.ops.aten.view.default(mm_98, [4, 64, 768]);  mm_98 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(view_222, torch.float32);  view_222 = None
        convert_element_type_371 = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_370, primals_40);  primals_40 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, 768)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
        mul_217 = torch.ops.aten.mul.Tensor(mul_215, mul_42);  mul_215 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_42, sum_42);  sum_42 = None
        sub_65 = torch.ops.aten.sub.Tensor(mul_216, sum_41);  mul_216 = sum_41 = None
        sub_66 = torch.ops.aten.sub.Tensor(sub_65, mul_218);  sub_65 = mul_218 = None
        mul_219 = torch.ops.aten.mul.Tensor(div_14, sub_66);  div_14 = sub_66 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_42);  convert_element_type_370 = mul_42 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
        add_86 = torch.ops.aten.add.Tensor(add_85, mul_219);  add_85 = mul_219 = None
        convert_element_type_372 = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16)
        view_223 = torch.ops.aten.view.default(convert_element_type_372, [256, 768]);  convert_element_type_372 = None
        permute_221 = torch.ops.aten.permute.default(view_223, [1, 0])
        mm_99 = torch.ops.aten.mm.default(permute_221, view_70);  permute_221 = view_70 = None
        mm_100 = torch.ops.aten.mm.default(view_223, permute_223);  view_223 = permute_223 = None
        view_224 = torch.ops.aten.view.default(mm_100, [4, 64, 3072]);  mm_100 = None
        convert_element_type_377 = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        view_69 = torch.ops.aten.view.default(mm_22, [4, 64, 3072]);  mm_22 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476)
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_29 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_222 = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_91, convert_element_type_91)
        mul_224 = torch.ops.aten.mul.Tensor(mul_223, -0.5);  mul_223 = None
        exp_8 = torch.ops.aten.exp.default(mul_224);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_226 = torch.ops.aten.mul.Tensor(convert_element_type_91, mul_225);  convert_element_type_91 = mul_225 = None
        add_88 = torch.ops.aten.add.Tensor(mul_222, mul_226);  mul_222 = mul_226 = None
        mul_227 = torch.ops.aten.mul.Tensor(convert_element_type_378, add_88);  convert_element_type_378 = add_88 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        view_225 = torch.ops.aten.view.default(convert_element_type_380, [256, 3072]);  convert_element_type_380 = None
        permute_225 = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_101 = torch.ops.aten.mm.default(permute_225, view_68);  permute_225 = view_68 = None
        mm_102 = torch.ops.aten.mm.default(view_225, permute_227);  view_225 = permute_227 = None
        view_226 = torch.ops.aten.view.default(mm_102, [4, 64, 768]);  mm_102 = None
        convert_element_type_385 = torch.ops.prims.convert_element_type.default(view_226, torch.float32);  view_226 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        mul_229 = torch.ops.aten.mul.Tensor(convert_element_type_385, primals_37);  primals_37 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, 768)
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        mul_231 = torch.ops.aten.mul.Tensor(mul_229, mul_37);  mul_229 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_37, sum_45);  sum_45 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_230, sum_44);  mul_230 = sum_44 = None
        sub_69 = torch.ops.aten.sub.Tensor(sub_68, mul_232);  sub_68 = mul_232 = None
        mul_233 = torch.ops.aten.mul.Tensor(div_15, sub_69);  div_15 = sub_69 = None
        mul_234 = torch.ops.aten.mul.Tensor(convert_element_type_385, mul_37);  convert_element_type_385 = mul_37 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        add_89 = torch.ops.aten.add.Tensor(add_86, mul_233);  add_86 = mul_233 = None
        convert_element_type_387 = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        view_227 = torch.ops.aten.view.default(convert_element_type_387, [256, 768]);  convert_element_type_387 = None
        permute_229 = torch.ops.aten.permute.default(view_227, [1, 0])
        permute_44 = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_65 = torch.ops.aten.view.default(permute_44, [4, 64, 768]);  permute_44 = None
        view_66 = torch.ops.aten.view.default(view_65, [256, 768]);  view_65 = None
        mm_103 = torch.ops.aten.mm.default(permute_229, view_66);  permute_229 = view_66 = None
        mm_104 = torch.ops.aten.mm.default(view_227, permute_231);  view_227 = permute_231 = None
        view_228 = torch.ops.aten.view.default(mm_104, [4, 64, 768]);  mm_104 = None
        convert_element_type_392 = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        view_229 = torch.ops.aten.view.default(view_228, [4, 64, 12, 64]);  view_228 = None
        permute_233 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_233, permute_42, permute_41, permute_43, getitem_85, getitem_86, None, None, 64, 64, 0.0, True, getitem_91, getitem_92, scale = 0.125);  permute_233 = permute_42 = permute_41 = permute_43 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_212 = _scaled_dot_product_flash_attention_backward_6[0]
        getitem_213 = _scaled_dot_product_flash_attention_backward_6[1]
        getitem_214 = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
        permute_234 = torch.ops.aten.permute.default(getitem_214, [0, 2, 1, 3]);  getitem_214 = None
        view_230 = torch.ops.aten.view.default(permute_234, [4, 64, 768]);  permute_234 = None
        permute_235 = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3]);  getitem_212 = None
        view_231 = torch.ops.aten.view.default(permute_235, [4, 64, 768]);  permute_235 = None
        permute_236 = torch.ops.aten.permute.default(getitem_213, [0, 2, 1, 3]);  getitem_213 = None
        view_232 = torch.ops.aten.view.default(permute_236, [4, 64, 768]);  permute_236 = None
        cat_6 = torch.ops.aten.cat.default([view_231, view_232, view_230], 2);  view_231 = view_232 = view_230 = None
        view_233 = torch.ops.aten.view.default(cat_6, [256, 2304]);  cat_6 = None
        permute_237 = torch.ops.aten.permute.default(view_233, [1, 0])
        mm_105 = torch.ops.aten.mm.default(permute_237, view_60);  permute_237 = view_60 = None
        mm_106 = torch.ops.aten.mm.default(view_233, permute_239);  view_233 = permute_239 = None
        view_234 = torch.ops.aten.view.default(mm_106, [4, 64, 768]);  mm_106 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        mul_236 = torch.ops.aten.mul.Tensor(convert_element_type_397, primals_34);  primals_34 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238 = torch.ops.aten.mul.Tensor(mul_236, mul_35);  mul_236 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_35, sum_48);  sum_48 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_237, sum_47);  mul_237 = sum_47 = None
        sub_72 = torch.ops.aten.sub.Tensor(sub_71, mul_239);  sub_71 = mul_239 = None
        mul_240 = torch.ops.aten.mul.Tensor(div_16, sub_72);  div_16 = sub_72 = None
        mul_241 = torch.ops.aten.mul.Tensor(convert_element_type_397, mul_35);  convert_element_type_397 = mul_35 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, mul_240);  add_89 = mul_240 = None
        convert_element_type_399 = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16)
        view_235 = torch.ops.aten.view.default(convert_element_type_399, [256, 768]);  convert_element_type_399 = None
        permute_241 = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_107 = torch.ops.aten.mm.default(permute_241, view_58);  permute_241 = view_58 = None
        mm_108 = torch.ops.aten.mm.default(view_235, permute_243);  view_235 = permute_243 = None
        view_236 = torch.ops.aten.view.default(mm_108, [4, 64, 3072]);  mm_108 = None
        convert_element_type_404 = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_405 = torch.ops.prims.convert_element_type.default(view_236, torch.float32);  view_236 = None
        view_57 = torch.ops.aten.view.default(mm_18, [4, 64, 3072]);  mm_18 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476)
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_24 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_243 = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_244 = torch.ops.aten.mul.Tensor(convert_element_type_75, convert_element_type_75)
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, -0.5);  mul_244 = None
        exp_9 = torch.ops.aten.exp.default(mul_245);  mul_245 = None
        mul_246 = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_75, mul_246);  convert_element_type_75 = mul_246 = None
        add_92 = torch.ops.aten.add.Tensor(mul_243, mul_247);  mul_243 = mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_405, add_92);  convert_element_type_405 = add_92 = None
        convert_element_type_407 = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        view_237 = torch.ops.aten.view.default(convert_element_type_407, [256, 3072]);  convert_element_type_407 = None
        permute_245 = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_109 = torch.ops.aten.mm.default(permute_245, view_56);  permute_245 = view_56 = None
        mm_110 = torch.ops.aten.mm.default(view_237, permute_247);  view_237 = permute_247 = None
        view_238 = torch.ops.aten.view.default(mm_110, [4, 64, 768]);  mm_110 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        mul_250 = torch.ops.aten.mul.Tensor(convert_element_type_412, primals_31);  primals_31 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, 768)
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
        mul_252 = torch.ops.aten.mul.Tensor(mul_250, mul_30);  mul_250 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_30, sum_51);  sum_51 = None
        sub_74 = torch.ops.aten.sub.Tensor(mul_251, sum_50);  mul_251 = sum_50 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, mul_253);  sub_74 = mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(div_17, sub_75);  div_17 = sub_75 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_412, mul_30);  convert_element_type_412 = mul_30 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        add_93 = torch.ops.aten.add.Tensor(add_90, mul_254);  add_90 = mul_254 = None
        convert_element_type_414 = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_239 = torch.ops.aten.view.default(convert_element_type_414, [256, 768]);  convert_element_type_414 = None
        permute_249 = torch.ops.aten.permute.default(view_239, [1, 0])
        permute_36 = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_53 = torch.ops.aten.view.default(permute_36, [4, 64, 768]);  permute_36 = None
        view_54 = torch.ops.aten.view.default(view_53, [256, 768]);  view_53 = None
        mm_111 = torch.ops.aten.mm.default(permute_249, view_54);  permute_249 = view_54 = None
        mm_112 = torch.ops.aten.mm.default(view_239, permute_251);  view_239 = permute_251 = None
        view_240 = torch.ops.aten.view.default(mm_112, [4, 64, 768]);  mm_112 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        view_241 = torch.ops.aten.view.default(view_240, [4, 64, 12, 64]);  view_240 = None
        permute_253 = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_253, permute_34, permute_33, permute_35, getitem_69, getitem_70, None, None, 64, 64, 0.0, True, getitem_75, getitem_76, scale = 0.125);  permute_253 = permute_34 = permute_33 = permute_35 = getitem_69 = getitem_70 = getitem_75 = getitem_76 = None
        getitem_215 = _scaled_dot_product_flash_attention_backward_7[0]
        getitem_216 = _scaled_dot_product_flash_attention_backward_7[1]
        getitem_217 = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
        permute_254 = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None
        view_242 = torch.ops.aten.view.default(permute_254, [4, 64, 768]);  permute_254 = None
        permute_255 = torch.ops.aten.permute.default(getitem_215, [0, 2, 1, 3]);  getitem_215 = None
        view_243 = torch.ops.aten.view.default(permute_255, [4, 64, 768]);  permute_255 = None
        permute_256 = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_244 = torch.ops.aten.view.default(permute_256, [4, 64, 768]);  permute_256 = None
        cat_7 = torch.ops.aten.cat.default([view_243, view_244, view_242], 2);  view_243 = view_244 = view_242 = None
        view_245 = torch.ops.aten.view.default(cat_7, [256, 2304]);  cat_7 = None
        permute_257 = torch.ops.aten.permute.default(view_245, [1, 0])
        mm_113 = torch.ops.aten.mm.default(permute_257, view_48);  permute_257 = view_48 = None
        mm_114 = torch.ops.aten.mm.default(view_245, permute_259);  view_245 = permute_259 = None
        view_246 = torch.ops.aten.view.default(mm_114, [4, 64, 768]);  mm_114 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_425 = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        mul_257 = torch.ops.aten.mul.Tensor(convert_element_type_424, primals_28);  primals_28 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, 768)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
        mul_259 = torch.ops.aten.mul.Tensor(mul_257, mul_28);  mul_257 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_28, sum_54);  sum_54 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_258, sum_53);  mul_258 = sum_53 = None
        sub_78 = torch.ops.aten.sub.Tensor(sub_77, mul_260);  sub_77 = mul_260 = None
        mul_261 = torch.ops.aten.mul.Tensor(div_18, sub_78);  div_18 = sub_78 = None
        mul_262 = torch.ops.aten.mul.Tensor(convert_element_type_424, mul_28);  convert_element_type_424 = mul_28 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, mul_261);  add_93 = mul_261 = None
        convert_element_type_426 = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        view_247 = torch.ops.aten.view.default(convert_element_type_426, [256, 768]);  convert_element_type_426 = None
        permute_261 = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_115 = torch.ops.aten.mm.default(permute_261, view_46);  permute_261 = view_46 = None
        mm_116 = torch.ops.aten.mm.default(view_247, permute_263);  view_247 = permute_263 = None
        view_248 = torch.ops.aten.view.default(mm_116, [4, 64, 3072]);  mm_116 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None
        view_45 = torch.ops.aten.view.default(mm_14, [4, 64, 3072]);  mm_14 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_26 = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476)
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_19 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_264 = torch.ops.aten.mul.Tensor(add_19, 0.5);  add_19 = None
        mul_265 = torch.ops.aten.mul.Tensor(convert_element_type_59, convert_element_type_59)
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
        exp_10 = torch.ops.aten.exp.default(mul_266);  mul_266 = None
        mul_267 = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_268 = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_267);  convert_element_type_59 = mul_267 = None
        add_96 = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(convert_element_type_432, add_96);  convert_element_type_432 = add_96 = None
        convert_element_type_434 = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        view_249 = torch.ops.aten.view.default(convert_element_type_434, [256, 3072]);  convert_element_type_434 = None
        permute_265 = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_265, view_44);  permute_265 = view_44 = None
        mm_118 = torch.ops.aten.mm.default(view_249, permute_267);  view_249 = permute_267 = None
        view_250 = torch.ops.aten.view.default(mm_118, [4, 64, 768]);  mm_118 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        convert_element_type_440 = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        mul_271 = torch.ops.aten.mul.Tensor(convert_element_type_439, primals_25);  primals_25 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, 768)
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
        mul_273 = torch.ops.aten.mul.Tensor(mul_271, mul_23);  mul_271 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_23, sum_57);  sum_57 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_272, sum_56);  mul_272 = sum_56 = None
        sub_81 = torch.ops.aten.sub.Tensor(sub_80, mul_274);  sub_80 = mul_274 = None
        mul_275 = torch.ops.aten.mul.Tensor(div_19, sub_81);  div_19 = sub_81 = None
        mul_276 = torch.ops.aten.mul.Tensor(convert_element_type_439, mul_23);  convert_element_type_439 = mul_23 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
        add_97 = torch.ops.aten.add.Tensor(add_94, mul_275);  add_94 = mul_275 = None
        convert_element_type_441 = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        view_251 = torch.ops.aten.view.default(convert_element_type_441, [256, 768]);  convert_element_type_441 = None
        permute_269 = torch.ops.aten.permute.default(view_251, [1, 0])
        permute_28 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_41 = torch.ops.aten.view.default(permute_28, [4, 64, 768]);  permute_28 = None
        view_42 = torch.ops.aten.view.default(view_41, [256, 768]);  view_41 = None
        mm_119 = torch.ops.aten.mm.default(permute_269, view_42);  permute_269 = view_42 = None
        mm_120 = torch.ops.aten.mm.default(view_251, permute_271);  view_251 = permute_271 = None
        view_252 = torch.ops.aten.view.default(mm_120, [4, 64, 768]);  mm_120 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        view_253 = torch.ops.aten.view.default(view_252, [4, 64, 12, 64]);  view_252 = None
        permute_273 = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_273, permute_26, permute_25, permute_27, getitem_53, getitem_54, None, None, 64, 64, 0.0, True, getitem_59, getitem_60, scale = 0.125);  permute_273 = permute_26 = permute_25 = permute_27 = getitem_53 = getitem_54 = getitem_59 = getitem_60 = None
        getitem_218 = _scaled_dot_product_flash_attention_backward_8[0]
        getitem_219 = _scaled_dot_product_flash_attention_backward_8[1]
        getitem_220 = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
        permute_274 = torch.ops.aten.permute.default(getitem_220, [0, 2, 1, 3]);  getitem_220 = None
        view_254 = torch.ops.aten.view.default(permute_274, [4, 64, 768]);  permute_274 = None
        permute_275 = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        view_255 = torch.ops.aten.view.default(permute_275, [4, 64, 768]);  permute_275 = None
        permute_276 = torch.ops.aten.permute.default(getitem_219, [0, 2, 1, 3]);  getitem_219 = None
        view_256 = torch.ops.aten.view.default(permute_276, [4, 64, 768]);  permute_276 = None
        cat_8 = torch.ops.aten.cat.default([view_255, view_256, view_254], 2);  view_255 = view_256 = view_254 = None
        view_257 = torch.ops.aten.view.default(cat_8, [256, 2304]);  cat_8 = None
        permute_277 = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_121 = torch.ops.aten.mm.default(permute_277, view_36);  permute_277 = view_36 = None
        mm_122 = torch.ops.aten.mm.default(view_257, permute_279);  view_257 = permute_279 = None
        view_258 = torch.ops.aten.view.default(mm_122, [4, 64, 768]);  mm_122 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        mul_278 = torch.ops.aten.mul.Tensor(convert_element_type_451, primals_22);  primals_22 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, 768)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280 = torch.ops.aten.mul.Tensor(mul_278, mul_21);  mul_278 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_21, sum_60);  sum_60 = None
        sub_83 = torch.ops.aten.sub.Tensor(mul_279, sum_59);  mul_279 = sum_59 = None
        sub_84 = torch.ops.aten.sub.Tensor(sub_83, mul_281);  sub_83 = mul_281 = None
        mul_282 = torch.ops.aten.mul.Tensor(div_20, sub_84);  div_20 = sub_84 = None
        mul_283 = torch.ops.aten.mul.Tensor(convert_element_type_451, mul_21);  convert_element_type_451 = mul_21 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, mul_282);  add_97 = mul_282 = None
        convert_element_type_453 = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16)
        view_259 = torch.ops.aten.view.default(convert_element_type_453, [256, 768]);  convert_element_type_453 = None
        permute_281 = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_281, view_34);  permute_281 = view_34 = None
        mm_124 = torch.ops.aten.mm.default(view_259, permute_283);  view_259 = permute_283 = None
        view_260 = torch.ops.aten.view.default(mm_124, [4, 64, 3072]);  mm_124 = None
        convert_element_type_458 = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_459 = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None
        view_33 = torch.ops.aten.view.default(mm_10, [4, 64, 3072]);  mm_10 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476)
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_285 = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_286 = torch.ops.aten.mul.Tensor(convert_element_type_43, convert_element_type_43)
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
        exp_11 = torch.ops.aten.exp.default(mul_287);  mul_287 = None
        mul_288 = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_289 = torch.ops.aten.mul.Tensor(convert_element_type_43, mul_288);  convert_element_type_43 = mul_288 = None
        add_100 = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_459, add_100);  convert_element_type_459 = add_100 = None
        convert_element_type_461 = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        view_261 = torch.ops.aten.view.default(convert_element_type_461, [256, 3072]);  convert_element_type_461 = None
        permute_285 = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_285, view_32);  permute_285 = view_32 = None
        mm_126 = torch.ops.aten.mm.default(view_261, permute_287);  view_261 = permute_287 = None
        view_262 = torch.ops.aten.view.default(mm_126, [4, 64, 768]);  mm_126 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        convert_element_type_467 = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        mul_292 = torch.ops.aten.mul.Tensor(convert_element_type_466, primals_19);  primals_19 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, 768)
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
        mul_294 = torch.ops.aten.mul.Tensor(mul_292, mul_16);  mul_292 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_16, sum_63);  sum_63 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_293, sum_62);  mul_293 = sum_62 = None
        sub_87 = torch.ops.aten.sub.Tensor(sub_86, mul_295);  sub_86 = mul_295 = None
        mul_296 = torch.ops.aten.mul.Tensor(div_21, sub_87);  div_21 = sub_87 = None
        mul_297 = torch.ops.aten.mul.Tensor(convert_element_type_466, mul_16);  convert_element_type_466 = mul_16 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        add_101 = torch.ops.aten.add.Tensor(add_98, mul_296);  add_98 = mul_296 = None
        convert_element_type_468 = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16)
        view_263 = torch.ops.aten.view.default(convert_element_type_468, [256, 768]);  convert_element_type_468 = None
        permute_289 = torch.ops.aten.permute.default(view_263, [1, 0])
        permute_20 = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_29 = torch.ops.aten.view.default(permute_20, [4, 64, 768]);  permute_20 = None
        view_30 = torch.ops.aten.view.default(view_29, [256, 768]);  view_29 = None
        mm_127 = torch.ops.aten.mm.default(permute_289, view_30);  permute_289 = view_30 = None
        mm_128 = torch.ops.aten.mm.default(view_263, permute_291);  view_263 = permute_291 = None
        view_264 = torch.ops.aten.view.default(mm_128, [4, 64, 768]);  mm_128 = None
        convert_element_type_473 = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        view_265 = torch.ops.aten.view.default(view_264, [4, 64, 12, 64]);  view_264 = None
        permute_293 = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_293, permute_18, permute_17, permute_19, getitem_37, getitem_38, None, None, 64, 64, 0.0, True, getitem_43, getitem_44, scale = 0.125);  permute_293 = permute_18 = permute_17 = permute_19 = getitem_37 = getitem_38 = getitem_43 = getitem_44 = None
        getitem_221 = _scaled_dot_product_flash_attention_backward_9[0]
        getitem_222 = _scaled_dot_product_flash_attention_backward_9[1]
        getitem_223 = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
        permute_294 = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None
        view_266 = torch.ops.aten.view.default(permute_294, [4, 64, 768]);  permute_294 = None
        permute_295 = torch.ops.aten.permute.default(getitem_221, [0, 2, 1, 3]);  getitem_221 = None
        view_267 = torch.ops.aten.view.default(permute_295, [4, 64, 768]);  permute_295 = None
        permute_296 = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3]);  getitem_222 = None
        view_268 = torch.ops.aten.view.default(permute_296, [4, 64, 768]);  permute_296 = None
        cat_9 = torch.ops.aten.cat.default([view_267, view_268, view_266], 2);  view_267 = view_268 = view_266 = None
        view_269 = torch.ops.aten.view.default(cat_9, [256, 2304]);  cat_9 = None
        permute_297 = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_129 = torch.ops.aten.mm.default(permute_297, view_24);  permute_297 = view_24 = None
        mm_130 = torch.ops.aten.mm.default(view_269, permute_299);  view_269 = permute_299 = None
        view_270 = torch.ops.aten.view.default(mm_130, [4, 64, 768]);  mm_130 = None
        convert_element_type_478 = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        mul_299 = torch.ops.aten.mul.Tensor(convert_element_type_478, primals_16);  primals_16 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, 768)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301 = torch.ops.aten.mul.Tensor(mul_299, mul_14);  mul_299 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_14, sum_66);  sum_66 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_300, sum_65);  mul_300 = sum_65 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, mul_302);  sub_89 = mul_302 = None
        mul_303 = torch.ops.aten.mul.Tensor(div_22, sub_90);  div_22 = sub_90 = None
        mul_304 = torch.ops.aten.mul.Tensor(convert_element_type_478, mul_14);  convert_element_type_478 = mul_14 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        add_102 = torch.ops.aten.add.Tensor(add_101, mul_303);  add_101 = mul_303 = None
        convert_element_type_480 = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16)
        view_271 = torch.ops.aten.view.default(convert_element_type_480, [256, 768]);  convert_element_type_480 = None
        permute_301 = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_301, view_22);  permute_301 = view_22 = None
        mm_132 = torch.ops.aten.mm.default(view_271, permute_303);  view_271 = permute_303 = None
        view_272 = torch.ops.aten.view.default(mm_132, [4, 64, 3072]);  mm_132 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_486 = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        view_21 = torch.ops.aten.view.default(mm_6, [4, 64, 3072]);  mm_6 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_306 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_27, convert_element_type_27)
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, -0.5);  mul_307 = None
        exp_12 = torch.ops.aten.exp.default(mul_308);  mul_308 = None
        mul_309 = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_310 = torch.ops.aten.mul.Tensor(convert_element_type_27, mul_309);  convert_element_type_27 = mul_309 = None
        add_104 = torch.ops.aten.add.Tensor(mul_306, mul_310);  mul_306 = mul_310 = None
        mul_311 = torch.ops.aten.mul.Tensor(convert_element_type_486, add_104);  convert_element_type_486 = add_104 = None
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        view_273 = torch.ops.aten.view.default(convert_element_type_488, [256, 3072]);  convert_element_type_488 = None
        permute_305 = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_305, view_20);  permute_305 = view_20 = None
        mm_134 = torch.ops.aten.mm.default(view_273, permute_307);  view_273 = permute_307 = None
        view_274 = torch.ops.aten.view.default(mm_134, [4, 64, 768]);  mm_134 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        convert_element_type_494 = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        mul_313 = torch.ops.aten.mul.Tensor(convert_element_type_493, primals_13);  primals_13 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, 768)
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
        mul_315 = torch.ops.aten.mul.Tensor(mul_313, mul_9);  mul_313 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_9, sum_69);  sum_69 = None
        sub_92 = torch.ops.aten.sub.Tensor(mul_314, sum_68);  mul_314 = sum_68 = None
        sub_93 = torch.ops.aten.sub.Tensor(sub_92, mul_316);  sub_92 = mul_316 = None
        mul_317 = torch.ops.aten.mul.Tensor(div_23, sub_93);  div_23 = sub_93 = None
        mul_318 = torch.ops.aten.mul.Tensor(convert_element_type_493, mul_9);  convert_element_type_493 = mul_9 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
        add_105 = torch.ops.aten.add.Tensor(add_102, mul_317);  add_102 = mul_317 = None
        convert_element_type_495 = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16)
        view_275 = torch.ops.aten.view.default(convert_element_type_495, [256, 768]);  convert_element_type_495 = None
        permute_309 = torch.ops.aten.permute.default(view_275, [1, 0])
        permute_12 = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_17 = torch.ops.aten.view.default(permute_12, [4, 64, 768]);  permute_12 = None
        view_18 = torch.ops.aten.view.default(view_17, [256, 768]);  view_17 = None
        mm_135 = torch.ops.aten.mm.default(permute_309, view_18);  permute_309 = view_18 = None
        mm_136 = torch.ops.aten.mm.default(view_275, permute_311);  view_275 = permute_311 = None
        view_276 = torch.ops.aten.view.default(mm_136, [4, 64, 768]);  mm_136 = None
        convert_element_type_500 = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        view_277 = torch.ops.aten.view.default(view_276, [4, 64, 12, 64]);  view_276 = None
        permute_313 = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_313, permute_10, permute_9, permute_11, getitem_21, getitem_22, None, None, 64, 64, 0.0, True, getitem_27, getitem_28, scale = 0.125);  permute_313 = permute_10 = permute_9 = permute_11 = getitem_21 = getitem_22 = getitem_27 = getitem_28 = None
        getitem_224 = _scaled_dot_product_flash_attention_backward_10[0]
        getitem_225 = _scaled_dot_product_flash_attention_backward_10[1]
        getitem_226 = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
        permute_314 = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3]);  getitem_226 = None
        view_278 = torch.ops.aten.view.default(permute_314, [4, 64, 768]);  permute_314 = None
        permute_315 = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None
        view_279 = torch.ops.aten.view.default(permute_315, [4, 64, 768]);  permute_315 = None
        permute_316 = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None
        view_280 = torch.ops.aten.view.default(permute_316, [4, 64, 768]);  permute_316 = None
        cat_10 = torch.ops.aten.cat.default([view_279, view_280, view_278], 2);  view_279 = view_280 = view_278 = None
        view_281 = torch.ops.aten.view.default(cat_10, [256, 2304]);  cat_10 = None
        permute_317 = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_317, view_12);  permute_317 = view_12 = None
        mm_138 = torch.ops.aten.mm.default(view_281, permute_319);  view_281 = permute_319 = None
        view_282 = torch.ops.aten.view.default(mm_138, [4, 64, 768]);  mm_138 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        convert_element_type_506 = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        mul_320 = torch.ops.aten.mul.Tensor(convert_element_type_505, primals_10);  primals_10 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, 768)
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        mul_322 = torch.ops.aten.mul.Tensor(mul_320, mul_7);  mul_320 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_7, sum_72);  sum_72 = None
        sub_95 = torch.ops.aten.sub.Tensor(mul_321, sum_71);  mul_321 = sum_71 = None
        sub_96 = torch.ops.aten.sub.Tensor(sub_95, mul_323);  sub_95 = mul_323 = None
        mul_324 = torch.ops.aten.mul.Tensor(div_24, sub_96);  div_24 = sub_96 = None
        mul_325 = torch.ops.aten.mul.Tensor(convert_element_type_505, mul_7);  convert_element_type_505 = mul_7 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        add_106 = torch.ops.aten.add.Tensor(add_105, mul_324);  add_105 = mul_324 = None
        convert_element_type_507 = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16)
        view_283 = torch.ops.aten.view.default(convert_element_type_507, [256, 768]);  convert_element_type_507 = None
        permute_321 = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_321, view_10);  permute_321 = view_10 = None
        mm_140 = torch.ops.aten.mm.default(view_283, permute_323);  view_283 = permute_323 = None
        view_284 = torch.ops.aten.view.default(mm_140, [4, 64, 3072]);  mm_140 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_513 = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        view_9 = torch.ops.aten.view.default(mm_2, [4, 64, 3072]);  mm_2 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_327 = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_328 = torch.ops.aten.mul.Tensor(convert_element_type_11, convert_element_type_11)
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, -0.5);  mul_328 = None
        exp_13 = torch.ops.aten.exp.default(mul_329);  mul_329 = None
        mul_330 = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_331 = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_330);  convert_element_type_11 = mul_330 = None
        add_108 = torch.ops.aten.add.Tensor(mul_327, mul_331);  mul_327 = mul_331 = None
        mul_332 = torch.ops.aten.mul.Tensor(convert_element_type_513, add_108);  convert_element_type_513 = add_108 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        view_285 = torch.ops.aten.view.default(convert_element_type_515, [256, 3072]);  convert_element_type_515 = None
        permute_325 = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_325, view_8);  permute_325 = view_8 = None
        mm_142 = torch.ops.aten.mm.default(view_285, permute_327);  view_285 = permute_327 = None
        view_286 = torch.ops.aten.view.default(mm_142, [4, 64, 768]);  mm_142 = None
        convert_element_type_520 = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        mul_334 = torch.ops.aten.mul.Tensor(convert_element_type_520, primals_7);  primals_7 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, 768)
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_334, [2], True)
        mul_336 = torch.ops.aten.mul.Tensor(mul_334, mul_2);  mul_334 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_336, [2], True);  mul_336 = None
        mul_337 = torch.ops.aten.mul.Tensor(mul_2, sum_75);  sum_75 = None
        sub_98 = torch.ops.aten.sub.Tensor(mul_335, sum_74);  mul_335 = sum_74 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, mul_337);  sub_98 = mul_337 = None
        mul_338 = torch.ops.aten.mul.Tensor(div_25, sub_99);  div_25 = sub_99 = None
        mul_339 = torch.ops.aten.mul.Tensor(convert_element_type_520, mul_2);  convert_element_type_520 = mul_2 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1]);  mul_339 = None
        add_109 = torch.ops.aten.add.Tensor(add_106, mul_338);  add_106 = mul_338 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        view_287 = torch.ops.aten.view.default(convert_element_type_522, [256, 768]);  convert_element_type_522 = None
        permute_329 = torch.ops.aten.permute.default(view_287, [1, 0])
        permute_4 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_5 = torch.ops.aten.view.default(permute_4, [4, 64, 768]);  permute_4 = None
        view_6 = torch.ops.aten.view.default(view_5, [256, 768]);  view_5 = None
        mm_143 = torch.ops.aten.mm.default(permute_329, view_6);  permute_329 = view_6 = None
        mm_144 = torch.ops.aten.mm.default(view_287, permute_331);  view_287 = permute_331 = None
        view_288 = torch.ops.aten.view.default(mm_144, [4, 64, 768]);  mm_144 = None
        convert_element_type_527 = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        view_289 = torch.ops.aten.view.default(view_288, [4, 64, 12, 64]);  view_288 = None
        permute_333 = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_333, permute_2, permute_1, permute_3, getitem_5, getitem_6, None, None, 64, 64, 0.0, True, getitem_11, getitem_12, scale = 0.125);  permute_333 = permute_2 = permute_1 = permute_3 = getitem_5 = getitem_6 = getitem_11 = getitem_12 = None
        getitem_227 = _scaled_dot_product_flash_attention_backward_11[0]
        getitem_228 = _scaled_dot_product_flash_attention_backward_11[1]
        getitem_229 = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
        permute_334 = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None
        view_290 = torch.ops.aten.view.default(permute_334, [4, 64, 768]);  permute_334 = None
        permute_335 = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_291 = torch.ops.aten.view.default(permute_335, [4, 64, 768]);  permute_335 = None
        permute_336 = torch.ops.aten.permute.default(getitem_228, [0, 2, 1, 3]);  getitem_228 = None
        view_292 = torch.ops.aten.view.default(permute_336, [4, 64, 768]);  permute_336 = None
        cat_11 = torch.ops.aten.cat.default([view_291, view_292, view_290], 2);  view_291 = view_292 = view_290 = None
        view_293 = torch.ops.aten.view.default(cat_11, [256, 2304]);  cat_11 = None
        permute_337 = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_337, view);  permute_337 = view = None
        mm_146 = torch.ops.aten.mm.default(view_293, permute_339);  view_293 = permute_339 = None
        view_294 = torch.ops.aten.view.default(mm_146, [4, 64, 768]);  mm_146 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None
        convert_element_type_533 = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        mul_341 = torch.ops.aten.mul.Tensor(convert_element_type_532, primals_4);  primals_4 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_341, mul);  mul_341 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul, sum_78);  sum_78 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_342, sum_77);  mul_342 = sum_77 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, mul_344);  sub_101 = mul_344 = None
        div_26 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_345 = torch.ops.aten.mul.Tensor(div_26, sub_102);  div_26 = sub_102 = None
        mul_346 = torch.ops.aten.mul.Tensor(convert_element_type_532, mul);  convert_element_type_532 = mul = None
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        add_110 = torch.ops.aten.add.Tensor(add_109, mul_345);  add_109 = mul_345 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(add_110, [0], True, dtype = torch.float32)
        view_295 = torch.ops.aten.view.default(sum_80, [64, 768]);  sum_80 = None
        eq = torch.ops.aten.eq.Scalar(iota, -1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_4 = torch.ops.aten.where.self(unsqueeze_2, full_default_1, view_295);  unsqueeze_2 = view_295 = None
        full_default_6 = torch.ops.aten.full.default([64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default_6, [iota], where_4, True);  full_default_6 = iota = where_4 = None
        eq_1 = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5 = torch.ops.aten.where.self(unsqueeze_3, full_default_1, add_110);  unsqueeze_3 = full_default_1 = add_110 = None
        full_default_8 = torch.ops.aten.full.default([65, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(full_default_8, [primals_1], where_5, True);  full_default_8 = primals_1 = where_5 = None
        add_111 = torch.ops.aten.add.Tensor(convert_element_type_209, index_put_1);  convert_element_type_209 = index_put_1 = None
        return (None, add_111, index_put, sum_79, convert_element_type_533, convert_element_type_527, sum_76, convert_element_type_521, convert_element_type_512, sum_73, convert_element_type_506, convert_element_type_500, sum_70, convert_element_type_494, convert_element_type_485, sum_67, convert_element_type_479, convert_element_type_473, sum_64, convert_element_type_467, convert_element_type_458, sum_61, convert_element_type_452, convert_element_type_446, sum_58, convert_element_type_440, convert_element_type_431, sum_55, convert_element_type_425, convert_element_type_419, sum_52, convert_element_type_413, convert_element_type_404, sum_49, convert_element_type_398, convert_element_type_392, sum_46, convert_element_type_386, convert_element_type_377, sum_43, convert_element_type_371, convert_element_type_365, sum_40, convert_element_type_359, convert_element_type_350, sum_37, convert_element_type_344, convert_element_type_338, sum_34, convert_element_type_332, convert_element_type_323, sum_31, convert_element_type_317, convert_element_type_311, sum_28, convert_element_type_305, convert_element_type_296, sum_25, convert_element_type_290, convert_element_type_284, sum_22, convert_element_type_278, convert_element_type_269, sum_19, convert_element_type_263, convert_element_type_257, sum_16, convert_element_type_251, convert_element_type_242, sum_13, convert_element_type_236, convert_element_type_230, sum_10, convert_element_type_224, convert_element_type_215, sum_7, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 64), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768,), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # primals_10
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_13
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_16
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_19
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_22
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_25
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_28
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # primals_31
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_34
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # primals_37
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_40
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_43
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_46
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # primals_49
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_52
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # primals_55
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_58
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # primals_61
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_64
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_67
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_70
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # primals_73
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # primals_76
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf26, (4, 64), dtype=torch.int64, is_leaf=True)  # primals_77
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf27, (64,), dtype=torch.int64, is_leaf=True)  # iota
    buf28 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf28, (4, 64, 768), is_leaf=True)  # embedding
    buf29 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64, 768), is_leaf=True)  # embedding_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (4, 64, 1), is_leaf=True)  # getitem_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4, 64, 1), is_leaf=True)  # rsqrt
    buf32 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf32, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view
    buf33 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf33, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_1
    reader.tensor(buf33, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_2
    reader.tensor(buf33, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_3
    buf34 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf34, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_5
    buf35 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf35, (4, 12, 64), is_leaf=True)  # getitem_6
    buf36 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf36, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_11
    buf37 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf37, (), dtype=torch.uint64, is_leaf=True)  # getitem_12
    buf38 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf38, (4, 64, 768), is_leaf=True)  # mul_2
    buf39 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf39, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_8
    buf40 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf40, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf41 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf41, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_10
    buf42 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf42, (4, 64, 768), is_leaf=True)  # mul_7
    buf43 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf43, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_12
    buf44 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf44, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_9
    reader.tensor(buf44, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_10
    reader.tensor(buf44, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_11
    buf45 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf45, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_21
    buf46 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf46, (4, 12, 64), is_leaf=True)  # getitem_22
    buf47 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf47, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_27
    buf48 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf48, (), dtype=torch.uint64, is_leaf=True)  # getitem_28
    buf49 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf49, (4, 64, 768), is_leaf=True)  # mul_9
    buf50 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf50, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_20
    buf51 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf51, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_6
    buf52 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf52, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_22
    buf53 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf53, (4, 64, 768), is_leaf=True)  # mul_14
    buf54 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf54, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_24
    buf55 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf55, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_17
    reader.tensor(buf55, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_18
    reader.tensor(buf55, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_19
    buf56 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf56, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_37
    buf57 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf57, (4, 12, 64), is_leaf=True)  # getitem_38
    buf58 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf58, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_43
    buf59 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf59, (), dtype=torch.uint64, is_leaf=True)  # getitem_44
    buf60 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf60, (4, 64, 768), is_leaf=True)  # mul_16
    buf61 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf61, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_32
    buf62 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf62, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_10
    buf63 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf63, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_34
    buf64 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf64, (4, 64, 768), is_leaf=True)  # mul_21
    buf65 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf65, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_36
    buf66 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf66, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_25
    reader.tensor(buf66, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_26
    reader.tensor(buf66, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_27
    buf67 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf67, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_53
    buf68 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf68, (4, 12, 64), is_leaf=True)  # getitem_54
    buf69 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf69, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_59
    buf70 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf70, (), dtype=torch.uint64, is_leaf=True)  # getitem_60
    buf71 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf71, (4, 64, 768), is_leaf=True)  # mul_23
    buf72 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf72, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_44
    buf73 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf73, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_14
    buf74 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf74, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_46
    buf75 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf75, (4, 64, 768), is_leaf=True)  # mul_28
    buf76 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf76, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_48
    buf77 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf77, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_33
    reader.tensor(buf77, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_34
    reader.tensor(buf77, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_35
    buf78 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf78, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_69
    buf79 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf79, (4, 12, 64), is_leaf=True)  # getitem_70
    buf80 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf80, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_75
    buf81 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf81, (), dtype=torch.uint64, is_leaf=True)  # getitem_76
    buf82 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf82, (4, 64, 768), is_leaf=True)  # mul_30
    buf83 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf83, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_56
    buf84 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf84, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_18
    buf85 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf85, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_58
    buf86 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf86, (4, 64, 768), is_leaf=True)  # mul_35
    buf87 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf87, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_60
    buf88 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf88, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_41
    reader.tensor(buf88, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_42
    reader.tensor(buf88, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_43
    buf89 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf89, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_85
    buf90 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf90, (4, 12, 64), is_leaf=True)  # getitem_86
    buf91 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf91, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_91
    buf92 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf92, (), dtype=torch.uint64, is_leaf=True)  # getitem_92
    buf93 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf93, (4, 64, 768), is_leaf=True)  # mul_37
    buf94 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf94, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_68
    buf95 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf95, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_22
    buf96 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf96, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_70
    buf97 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf97, (4, 64, 768), is_leaf=True)  # mul_42
    buf98 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf98, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_72
    buf99 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_49
    reader.tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_50
    reader.tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_51
    buf100 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf100, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_101
    buf101 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (4, 12, 64), is_leaf=True)  # getitem_102
    buf102 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf102, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_107
    buf103 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf103, (), dtype=torch.uint64, is_leaf=True)  # getitem_108
    buf104 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf104, (4, 64, 768), is_leaf=True)  # mul_44
    buf105 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf105, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_80
    buf106 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf106, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_26
    buf107 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf107, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_82
    buf108 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf108, (4, 64, 768), is_leaf=True)  # mul_49
    buf109 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf109, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_84
    buf110 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf110, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_57
    reader.tensor(buf110, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_58
    reader.tensor(buf110, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_59
    buf111 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf111, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_117
    buf112 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf112, (4, 12, 64), is_leaf=True)  # getitem_118
    buf113 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf113, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_123
    buf114 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf114, (), dtype=torch.uint64, is_leaf=True)  # getitem_124
    buf115 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf115, (4, 64, 768), is_leaf=True)  # mul_51
    buf116 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf116, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_92
    buf117 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf117, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_30
    buf118 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf118, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_94
    buf119 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf119, (4, 64, 768), is_leaf=True)  # mul_56
    buf120 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf120, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_96
    buf121 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf121, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_65
    reader.tensor(buf121, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_66
    reader.tensor(buf121, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_67
    buf122 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf122, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_133
    buf123 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf123, (4, 12, 64), is_leaf=True)  # getitem_134
    buf124 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf124, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_139
    buf125 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf125, (), dtype=torch.uint64, is_leaf=True)  # getitem_140
    buf126 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf126, (4, 64, 768), is_leaf=True)  # mul_58
    buf127 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf127, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_104
    buf128 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf128, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_34
    buf129 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf129, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_106
    buf130 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf130, (4, 64, 768), is_leaf=True)  # mul_63
    buf131 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf131, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_108
    buf132 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf132, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_73
    reader.tensor(buf132, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_74
    reader.tensor(buf132, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_75
    buf133 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf133, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_149
    buf134 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf134, (4, 12, 64), is_leaf=True)  # getitem_150
    buf135 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf135, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_155
    buf136 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf136, (), dtype=torch.uint64, is_leaf=True)  # getitem_156
    buf137 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf137, (4, 64, 768), is_leaf=True)  # mul_65
    buf138 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf138, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_116
    buf139 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf139, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_38
    buf140 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf140, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_118
    buf141 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf141, (4, 64, 768), is_leaf=True)  # mul_70
    buf142 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf142, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_120
    buf143 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf143, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_81
    reader.tensor(buf143, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_82
    reader.tensor(buf143, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_83
    buf144 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf144, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_165
    buf145 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf145, (4, 12, 64), is_leaf=True)  # getitem_166
    buf146 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf146, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_171
    buf147 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf147, (), dtype=torch.uint64, is_leaf=True)  # getitem_172
    buf148 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf148, (4, 64, 768), is_leaf=True)  # mul_72
    buf149 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf149, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_128
    buf150 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf150, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_42
    buf151 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf151, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_130
    buf152 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf152, (4, 64, 768), is_leaf=True)  # mul_77
    buf153 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf153, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_132
    buf154 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf154, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=768, is_leaf=True)  # permute_89
    reader.tensor(buf154, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, is_leaf=True)  # permute_90
    reader.tensor(buf154, (4, 12, 64, 64), (147456, 64, 2304, 1), dtype=torch.bfloat16, storage_offset=1536, is_leaf=True)  # permute_91
    buf155 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf155, (4, 12, 64, 64), (49152, 64, 768, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_181
    buf156 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf156, (4, 12, 64), is_leaf=True)  # getitem_182
    buf157 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf157, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_187
    buf158 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf158, (), dtype=torch.uint64, is_leaf=True)  # getitem_188
    buf159 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf159, (4, 64, 768), is_leaf=True)  # mul_79
    buf160 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf160, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_140
    buf161 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf161, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # mm_46
    buf162 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf162, (256, 3072), dtype=torch.bfloat16, is_leaf=True)  # view_142
    buf163 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf163, (4, 64, 768), is_leaf=True)  # mul_84
    buf164 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf164, (256, 768), dtype=torch.bfloat16, is_leaf=True)  # view_144
    buf165 = reader.storage(None, 33280, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf165, (4, 64, 65), dtype=torch.bfloat16, is_leaf=True)  # view_145
    buf166 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256, 1), is_leaf=True)  # amax
    buf167 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf167, (256, 1), is_leaf=True)  # log
    buf168 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf168, (), is_leaf=True)  # convert_element_type_199
    buf169 = reader.storage(None, 99840, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf169, (65, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_99
    buf170 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf170, (4, 64, 1), is_leaf=True)  # div_2
    buf171 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf171, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_103
    buf172 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf172, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_107
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (4, 64, 1), is_leaf=True)  # div_3
    buf174 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf174, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_111
    buf175 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf175, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_119
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4, 64, 1), is_leaf=True)  # div_4
    buf177 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf177, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_123
    buf178 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf178, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_127
    buf179 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf179, (4, 64, 1), is_leaf=True)  # div_5
    buf180 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf180, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_131
    buf181 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf181, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_139
    buf182 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf182, (4, 64, 1), is_leaf=True)  # div_6
    buf183 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf183, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_143
    buf184 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf184, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_147
    buf185 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf185, (4, 64, 1), is_leaf=True)  # div_7
    buf186 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf186, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_151
    buf187 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf187, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_159
    buf188 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf188, (4, 64, 1), is_leaf=True)  # div_8
    buf189 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf189, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_163
    buf190 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf190, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_167
    buf191 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf191, (4, 64, 1), is_leaf=True)  # div_9
    buf192 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf192, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_171
    buf193 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf193, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_179
    buf194 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4, 64, 1), is_leaf=True)  # div_10
    buf195 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf195, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_183
    buf196 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf196, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_187
    buf197 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf197, (4, 64, 1), is_leaf=True)  # div_11
    buf198 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf198, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_191
    buf199 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf199, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_199
    buf200 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf200, (4, 64, 1), is_leaf=True)  # div_12
    buf201 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf201, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_203
    buf202 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf202, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_207
    buf203 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf203, (4, 64, 1), is_leaf=True)  # div_13
    buf204 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf204, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_211
    buf205 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf205, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_219
    buf206 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf206, (4, 64, 1), is_leaf=True)  # div_14
    buf207 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf207, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_223
    buf208 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf208, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_227
    buf209 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf209, (4, 64, 1), is_leaf=True)  # div_15
    buf210 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf210, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_231
    buf211 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf211, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_239
    buf212 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf212, (4, 64, 1), is_leaf=True)  # div_16
    buf213 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf213, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_243
    buf214 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf214, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_247
    buf215 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf215, (4, 64, 1), is_leaf=True)  # div_17
    buf216 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf216, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_251
    buf217 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf217, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_259
    buf218 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf218, (4, 64, 1), is_leaf=True)  # div_18
    buf219 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf219, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_263
    buf220 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf220, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_267
    buf221 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf221, (4, 64, 1), is_leaf=True)  # div_19
    buf222 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf222, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_271
    buf223 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf223, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_279
    buf224 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf224, (4, 64, 1), is_leaf=True)  # div_20
    buf225 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf225, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_283
    buf226 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf226, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_287
    buf227 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf227, (4, 64, 1), is_leaf=True)  # div_21
    buf228 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf228, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_291
    buf229 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf229, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_299
    buf230 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf230, (4, 64, 1), is_leaf=True)  # div_22
    buf231 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf231, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_303
    buf232 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf232, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_307
    buf233 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf233, (4, 64, 1), is_leaf=True)  # div_23
    buf234 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf234, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_311
    buf235 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf235, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_319
    buf236 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf236, (4, 64, 1), is_leaf=True)  # div_24
    buf237 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf237, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # permute_323
    buf238 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf238, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_327
    buf239 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf239, (4, 64, 1), is_leaf=True)  # div_25
    buf240 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf240, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_331
    buf241 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf241, (2304, 768), dtype=torch.bfloat16, is_leaf=True)  # permute_339
    buf242 = reader.storage(None, 33280, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf242, (4, 64, 65), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
    buf243 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf243, (), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)