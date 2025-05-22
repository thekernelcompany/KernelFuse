class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 64]", primals_4: "f32[768]", primals_7: "f32[768]", primals_10: "f32[768]", primals_13: "f32[768]", primals_16: "f32[768]", primals_19: "f32[768]", primals_22: "f32[768]", primals_25: "f32[768]", primals_28: "f32[768]", primals_31: "f32[768]", primals_34: "f32[768]", primals_37: "f32[768]", primals_40: "f32[768]", primals_43: "f32[768]", primals_46: "f32[768]", primals_49: "f32[768]", primals_52: "f32[768]", primals_55: "f32[768]", primals_58: "f32[768]", primals_61: "f32[768]", primals_64: "f32[768]", primals_67: "f32[768]", primals_70: "f32[768]", primals_73: "f32[768]", primals_76: "f32[768]", primals_77: "i64[4, 64]", iota: "i64[64]", embedding: "f32[4, 64, 768]", embedding_1: "f32[64, 768]", getitem_1: "f32[4, 64, 1]", rsqrt: "f32[4, 64, 1]", view: "bf16[256, 768]", permute_1: "bf16[4, 12, 64, 64]", permute_2: "bf16[4, 12, 64, 64]", permute_3: "bf16[4, 12, 64, 64]", getitem_5: "bf16[4, 12, 64, 64]", getitem_6: "f32[4, 12, 64]", getitem_11: "u64[2]", getitem_12: "u64[]", mul_2: "f32[4, 64, 768]", view_8: "bf16[256, 768]", mm_2: "bf16[256, 3072]", view_10: "bf16[256, 3072]", mul_7: "f32[4, 64, 768]", view_12: "bf16[256, 768]", permute_9: "bf16[4, 12, 64, 64]", permute_10: "bf16[4, 12, 64, 64]", permute_11: "bf16[4, 12, 64, 64]", getitem_21: "bf16[4, 12, 64, 64]", getitem_22: "f32[4, 12, 64]", getitem_27: "u64[2]", getitem_28: "u64[]", mul_9: "f32[4, 64, 768]", view_20: "bf16[256, 768]", mm_6: "bf16[256, 3072]", view_22: "bf16[256, 3072]", mul_14: "f32[4, 64, 768]", view_24: "bf16[256, 768]", permute_17: "bf16[4, 12, 64, 64]", permute_18: "bf16[4, 12, 64, 64]", permute_19: "bf16[4, 12, 64, 64]", getitem_37: "bf16[4, 12, 64, 64]", getitem_38: "f32[4, 12, 64]", getitem_43: "u64[2]", getitem_44: "u64[]", mul_16: "f32[4, 64, 768]", view_32: "bf16[256, 768]", mm_10: "bf16[256, 3072]", view_34: "bf16[256, 3072]", mul_21: "f32[4, 64, 768]", view_36: "bf16[256, 768]", permute_25: "bf16[4, 12, 64, 64]", permute_26: "bf16[4, 12, 64, 64]", permute_27: "bf16[4, 12, 64, 64]", getitem_53: "bf16[4, 12, 64, 64]", getitem_54: "f32[4, 12, 64]", getitem_59: "u64[2]", getitem_60: "u64[]", mul_23: "f32[4, 64, 768]", view_44: "bf16[256, 768]", mm_14: "bf16[256, 3072]", view_46: "bf16[256, 3072]", mul_28: "f32[4, 64, 768]", view_48: "bf16[256, 768]", permute_33: "bf16[4, 12, 64, 64]", permute_34: "bf16[4, 12, 64, 64]", permute_35: "bf16[4, 12, 64, 64]", getitem_69: "bf16[4, 12, 64, 64]", getitem_70: "f32[4, 12, 64]", getitem_75: "u64[2]", getitem_76: "u64[]", mul_30: "f32[4, 64, 768]", view_56: "bf16[256, 768]", mm_18: "bf16[256, 3072]", view_58: "bf16[256, 3072]", mul_35: "f32[4, 64, 768]", view_60: "bf16[256, 768]", permute_41: "bf16[4, 12, 64, 64]", permute_42: "bf16[4, 12, 64, 64]", permute_43: "bf16[4, 12, 64, 64]", getitem_85: "bf16[4, 12, 64, 64]", getitem_86: "f32[4, 12, 64]", getitem_91: "u64[2]", getitem_92: "u64[]", mul_37: "f32[4, 64, 768]", view_68: "bf16[256, 768]", mm_22: "bf16[256, 3072]", view_70: "bf16[256, 3072]", mul_42: "f32[4, 64, 768]", view_72: "bf16[256, 768]", permute_49: "bf16[4, 12, 64, 64]", permute_50: "bf16[4, 12, 64, 64]", permute_51: "bf16[4, 12, 64, 64]", getitem_101: "bf16[4, 12, 64, 64]", getitem_102: "f32[4, 12, 64]", getitem_107: "u64[2]", getitem_108: "u64[]", mul_44: "f32[4, 64, 768]", view_80: "bf16[256, 768]", mm_26: "bf16[256, 3072]", view_82: "bf16[256, 3072]", mul_49: "f32[4, 64, 768]", view_84: "bf16[256, 768]", permute_57: "bf16[4, 12, 64, 64]", permute_58: "bf16[4, 12, 64, 64]", permute_59: "bf16[4, 12, 64, 64]", getitem_117: "bf16[4, 12, 64, 64]", getitem_118: "f32[4, 12, 64]", getitem_123: "u64[2]", getitem_124: "u64[]", mul_51: "f32[4, 64, 768]", view_92: "bf16[256, 768]", mm_30: "bf16[256, 3072]", view_94: "bf16[256, 3072]", mul_56: "f32[4, 64, 768]", view_96: "bf16[256, 768]", permute_65: "bf16[4, 12, 64, 64]", permute_66: "bf16[4, 12, 64, 64]", permute_67: "bf16[4, 12, 64, 64]", getitem_133: "bf16[4, 12, 64, 64]", getitem_134: "f32[4, 12, 64]", getitem_139: "u64[2]", getitem_140: "u64[]", mul_58: "f32[4, 64, 768]", view_104: "bf16[256, 768]", mm_34: "bf16[256, 3072]", view_106: "bf16[256, 3072]", mul_63: "f32[4, 64, 768]", view_108: "bf16[256, 768]", permute_73: "bf16[4, 12, 64, 64]", permute_74: "bf16[4, 12, 64, 64]", permute_75: "bf16[4, 12, 64, 64]", getitem_149: "bf16[4, 12, 64, 64]", getitem_150: "f32[4, 12, 64]", getitem_155: "u64[2]", getitem_156: "u64[]", mul_65: "f32[4, 64, 768]", view_116: "bf16[256, 768]", mm_38: "bf16[256, 3072]", view_118: "bf16[256, 3072]", mul_70: "f32[4, 64, 768]", view_120: "bf16[256, 768]", permute_81: "bf16[4, 12, 64, 64]", permute_82: "bf16[4, 12, 64, 64]", permute_83: "bf16[4, 12, 64, 64]", getitem_165: "bf16[4, 12, 64, 64]", getitem_166: "f32[4, 12, 64]", getitem_171: "u64[2]", getitem_172: "u64[]", mul_72: "f32[4, 64, 768]", view_128: "bf16[256, 768]", mm_42: "bf16[256, 3072]", view_130: "bf16[256, 3072]", mul_77: "f32[4, 64, 768]", view_132: "bf16[256, 768]", permute_89: "bf16[4, 12, 64, 64]", permute_90: "bf16[4, 12, 64, 64]", permute_91: "bf16[4, 12, 64, 64]", getitem_181: "bf16[4, 12, 64, 64]", getitem_182: "f32[4, 12, 64]", getitem_187: "u64[2]", getitem_188: "u64[]", mul_79: "f32[4, 64, 768]", view_140: "bf16[256, 768]", mm_46: "bf16[256, 3072]", view_142: "bf16[256, 3072]", mul_84: "f32[4, 64, 768]", view_144: "bf16[256, 768]", view_145: "bf16[4, 64, 65]", amax: "f32[256, 1]", log: "f32[256, 1]", convert_element_type_199: "f32[]", permute_99: "bf16[65, 768]", div_2: "f32[4, 64, 1]", permute_103: "bf16[768, 3072]", permute_107: "bf16[3072, 768]", div_3: "f32[4, 64, 1]", permute_111: "bf16[768, 768]", permute_119: "bf16[2304, 768]", div_4: "f32[4, 64, 1]", permute_123: "bf16[768, 3072]", permute_127: "bf16[3072, 768]", div_5: "f32[4, 64, 1]", permute_131: "bf16[768, 768]", permute_139: "bf16[2304, 768]", div_6: "f32[4, 64, 1]", permute_143: "bf16[768, 3072]", permute_147: "bf16[3072, 768]", div_7: "f32[4, 64, 1]", permute_151: "bf16[768, 768]", permute_159: "bf16[2304, 768]", div_8: "f32[4, 64, 1]", permute_163: "bf16[768, 3072]", permute_167: "bf16[3072, 768]", div_9: "f32[4, 64, 1]", permute_171: "bf16[768, 768]", permute_179: "bf16[2304, 768]", div_10: "f32[4, 64, 1]", permute_183: "bf16[768, 3072]", permute_187: "bf16[3072, 768]", div_11: "f32[4, 64, 1]", permute_191: "bf16[768, 768]", permute_199: "bf16[2304, 768]", div_12: "f32[4, 64, 1]", permute_203: "bf16[768, 3072]", permute_207: "bf16[3072, 768]", div_13: "f32[4, 64, 1]", permute_211: "bf16[768, 768]", permute_219: "bf16[2304, 768]", div_14: "f32[4, 64, 1]", permute_223: "bf16[768, 3072]", permute_227: "bf16[3072, 768]", div_15: "f32[4, 64, 1]", permute_231: "bf16[768, 768]", permute_239: "bf16[2304, 768]", div_16: "f32[4, 64, 1]", permute_243: "bf16[768, 3072]", permute_247: "bf16[3072, 768]", div_17: "f32[4, 64, 1]", permute_251: "bf16[768, 768]", permute_259: "bf16[2304, 768]", div_18: "f32[4, 64, 1]", permute_263: "bf16[768, 3072]", permute_267: "bf16[3072, 768]", div_19: "f32[4, 64, 1]", permute_271: "bf16[768, 768]", permute_279: "bf16[2304, 768]", div_20: "f32[4, 64, 1]", permute_283: "bf16[768, 3072]", permute_287: "bf16[3072, 768]", div_21: "f32[4, 64, 1]", permute_291: "bf16[768, 768]", permute_299: "bf16[2304, 768]", div_22: "f32[4, 64, 1]", permute_303: "bf16[768, 3072]", permute_307: "bf16[3072, 768]", div_23: "f32[4, 64, 1]", permute_311: "bf16[768, 768]", permute_319: "bf16[2304, 768]", div_24: "f32[4, 64, 1]", permute_323: "bf16[768, 3072]", permute_327: "bf16[3072, 768]", div_25: "f32[4, 64, 1]", permute_331: "bf16[768, 768]", permute_339: "bf16[2304, 768]", tangents_1: "bf16[4, 64, 65]", tangents_2: "f32[]"):
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_2, convert_element_type_199);  tangents_2 = convert_element_type_199 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        view_147: "i64[256]" = torch.ops.aten.reshape.default(primals_77, [-1]);  primals_77 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        unsqueeze_1: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(view_147, 1);  view_147 = None
        ne_3: "b8[256, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -1)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[256, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
        scatter_upon_const_tensor: "f32[256, 65]" = torch__inductor_fx_passes_post_grad_scatter_upon_const_tensor(shape = [256, 65], background_val = 0, dtype = torch.float32, dim = 1, selector = where_2, val = -1.0);  where_2 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[256, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = None
        mul_86: "f32[256, 65]" = torch.ops.aten.mul.Tensor(scatter_upon_const_tensor, where_3);  scatter_upon_const_tensor = where_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        view_146: "bf16[256, 65]" = torch.ops.aten.reshape.default(view_145, [-1, 65]);  view_145 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        convert_element_type_196: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        sub_25: "f32[256, 65]" = torch.ops.aten.sub.Tensor(convert_element_type_196, amax);  convert_element_type_196 = amax = None
        sub_26: "f32[256, 65]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_197: "bf16[256, 65]" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        convert_element_type_198: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(convert_element_type_197, torch.float32);  convert_element_type_197 = None
        exp_1: "f32[256, 65]" = torch.ops.aten.exp.default(convert_element_type_198);  convert_element_type_198 = None
        sum_4: "f32[256, 1]" = torch.ops.aten.sum.dim_IntList(mul_86, [1], True)
        mul_87: "f32[256, 65]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_27: "f32[256, 65]" = torch.ops.aten.sub.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
        convert_element_type_203: "bf16[256, 65]" = torch.ops.prims.convert_element_type.default(sub_27, torch.bfloat16);  sub_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        view_148: "bf16[4, 64, 65]" = torch.ops.aten.reshape.default(convert_element_type_203, [4, 64, 65]);  convert_element_type_203 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        add_62: "bf16[4, 64, 65]" = torch.ops.aten.add.Tensor(tangents_1, view_148);  tangents_1 = view_148 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:327 in forward, code: logits = self.lm_head(x)
        view_149: "bf16[256, 65]" = torch.ops.aten.reshape.default(add_62, [256, 65]);  add_62 = None
        permute_97: "bf16[65, 256]" = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_49: "bf16[65, 768]" = torch.ops.aten.mm.default(permute_97, view_144);  permute_97 = view_144 = None
        mm_50: "bf16[256, 768]" = torch.ops.aten.mm.default(view_149, permute_99);  view_149 = permute_99 = None
        view_150: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_50, [4, 64, 768]);  mm_50 = None
        convert_element_type_208: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        convert_element_type_209: "f32[65, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_89: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_208, primals_76);  primals_76 = None
        mul_90: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_89, 768)
        sum_5: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True)
        mul_91: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_89, mul_84);  mul_89 = None
        sum_6: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [2], True);  mul_91 = None
        mul_92: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_6);  sum_6 = None
        sub_29: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_90, sum_5);  mul_90 = sum_5 = None
        sub_30: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_92);  sub_29 = mul_92 = None
        mul_93: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_30);  div_2 = sub_30 = None
        mul_94: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_208, mul_84);  convert_element_type_208 = mul_84 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1]);  mul_94 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_210: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_151: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_210, [256, 768]);  convert_element_type_210 = None
        permute_101: "bf16[768, 256]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_51: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_101, view_142);  permute_101 = view_142 = None
        mm_52: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_151, permute_103);  view_151 = permute_103 = None
        view_152: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_52, [4, 64, 3072]);  mm_52 = None
        convert_element_type_215: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_216: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_141: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_46, [4, 64, 3072]);  mm_46 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_187: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_82: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476)
        erf_11: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_59: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_59, 0.5);  add_59 = None
        mul_97: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, convert_element_type_187)
        mul_98: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_97, -0.5);  mul_97 = None
        exp_2: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_98);  mul_98 = None
        mul_99: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_100: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, mul_99);  convert_element_type_187 = mul_99 = None
        add_64: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_96, mul_100);  mul_96 = mul_100 = None
        mul_101: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_216, add_64);  convert_element_type_216 = add_64 = None
        convert_element_type_218: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_153: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_218, [256, 3072]);  convert_element_type_218 = None
        permute_105: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_53: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_105, view_140);  permute_105 = view_140 = None
        mm_54: "bf16[256, 768]" = torch.ops.aten.mm.default(view_153, permute_107);  view_153 = permute_107 = None
        view_154: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_54, [4, 64, 768]);  mm_54 = None
        convert_element_type_223: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None
        convert_element_type_224: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_103: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_223, primals_73);  primals_73 = None
        mul_104: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_103, 768)
        sum_8: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True)
        mul_105: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_103, mul_79);  mul_103 = None
        sum_9: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True);  mul_105 = None
        mul_106: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_9);  sum_9 = None
        sub_32: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_104, sum_8);  mul_104 = sum_8 = None
        sub_33: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_106);  sub_32 = mul_106 = None
        mul_107: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_33);  div_3 = sub_33 = None
        mul_108: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_223, mul_79);  convert_element_type_223 = mul_79 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_108, [0, 1]);  mul_108 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_65: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(mul_93, mul_107);  mul_93 = mul_107 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_225: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_155: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_225, [256, 768]);  convert_element_type_225 = None
        permute_109: "bf16[768, 256]" = torch.ops.aten.permute.default(view_155, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_92: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_137: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_92, [4, 64, 768]);  permute_92 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_138: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_137, [256, 768]);  view_137 = None
        mm_55: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_109, view_138);  permute_109 = view_138 = None
        mm_56: "bf16[256, 768]" = torch.ops.aten.mm.default(view_155, permute_111);  view_155 = permute_111 = None
        view_156: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_56, [4, 64, 768]);  mm_56 = None
        convert_element_type_230: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_157: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_156, [4, 64, 12, 64]);  view_156 = None
        permute_113: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_113, permute_90, permute_89, permute_91, getitem_181, getitem_182, None, None, 64, 64, 0.0, True, getitem_187, getitem_188, scale = 0.125);  permute_113 = permute_90 = permute_89 = permute_91 = getitem_181 = getitem_182 = getitem_187 = getitem_188 = None
        getitem_194: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward[0]
        getitem_195: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward[1]
        getitem_196: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_114: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_158: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_114, [4, 64, 768]);  permute_114 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_115: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_159: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_115, [4, 64, 768]);  permute_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_116: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_160: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_116, [4, 64, 768]);  permute_116 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_159, view_160, view_158], 2);  view_159 = view_160 = view_158 = None
        view_161: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat, [256, 2304]);  cat = None
        permute_117: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_57: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_117, view_132);  permute_117 = view_132 = None
        mm_58: "bf16[256, 768]" = torch.ops.aten.mm.default(view_161, permute_119);  view_161 = permute_119 = None
        view_162: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_58, [4, 64, 768]);  mm_58 = None
        convert_element_type_235: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        convert_element_type_236: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_110: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_235, primals_70);  primals_70 = None
        mul_111: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_110, 768)
        sum_11: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True)
        mul_112: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_110, mul_77);  mul_110 = None
        sum_12: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [2], True);  mul_112 = None
        mul_113: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_12);  sum_12 = None
        sub_35: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_111, sum_11);  mul_111 = sum_11 = None
        sub_36: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_113);  sub_35 = mul_113 = None
        mul_114: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_36);  div_4 = sub_36 = None
        mul_115: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_235, mul_77);  convert_element_type_235 = mul_77 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1]);  mul_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_66: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_65, mul_114);  add_65 = mul_114 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_237: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_163: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_237, [256, 768]);  convert_element_type_237 = None
        permute_121: "bf16[768, 256]" = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_59: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_121, view_130);  permute_121 = view_130 = None
        mm_60: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_163, permute_123);  view_163 = permute_123 = None
        view_164: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_60, [4, 64, 3072]);  mm_60 = None
        convert_element_type_242: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_243: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_129: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_42, [4, 64, 3072]);  mm_42 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_171: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_75: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.7071067811865476)
        erf_10: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_117: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_118: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, convert_element_type_171)
        mul_119: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_118, -0.5);  mul_118 = None
        exp_3: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_119);  mul_119 = None
        mul_120: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_121: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, mul_120);  convert_element_type_171 = mul_120 = None
        add_68: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_117, mul_121);  mul_117 = mul_121 = None
        mul_122: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_243, add_68);  convert_element_type_243 = add_68 = None
        convert_element_type_245: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_165: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_245, [256, 3072]);  convert_element_type_245 = None
        permute_125: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_61: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_125, view_128);  permute_125 = view_128 = None
        mm_62: "bf16[256, 768]" = torch.ops.aten.mm.default(view_165, permute_127);  view_165 = permute_127 = None
        view_166: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_62, [4, 64, 768]);  mm_62 = None
        convert_element_type_250: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        convert_element_type_251: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_124: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_250, primals_67);  primals_67 = None
        mul_125: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_124, 768)
        sum_14: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True)
        mul_126: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_124, mul_72);  mul_124 = None
        sum_15: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [2], True);  mul_126 = None
        mul_127: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_15);  sum_15 = None
        sub_38: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_125, sum_14);  mul_125 = sum_14 = None
        sub_39: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_127);  sub_38 = mul_127 = None
        mul_128: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_39);  div_5 = sub_39 = None
        mul_129: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_250, mul_72);  convert_element_type_250 = mul_72 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_129, [0, 1]);  mul_129 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_69: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_66, mul_128);  add_66 = mul_128 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_252: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_167: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_252, [256, 768]);  convert_element_type_252 = None
        permute_129: "bf16[768, 256]" = torch.ops.aten.permute.default(view_167, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_84: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_125: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_84, [4, 64, 768]);  permute_84 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_126: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_125, [256, 768]);  view_125 = None
        mm_63: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_129, view_126);  permute_129 = view_126 = None
        mm_64: "bf16[256, 768]" = torch.ops.aten.mm.default(view_167, permute_131);  view_167 = permute_131 = None
        view_168: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_64, [4, 64, 768]);  mm_64 = None
        convert_element_type_257: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_169: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_168, [4, 64, 12, 64]);  view_168 = None
        permute_133: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_133, permute_82, permute_81, permute_83, getitem_165, getitem_166, None, None, 64, 64, 0.0, True, getitem_171, getitem_172, scale = 0.125);  permute_133 = permute_82 = permute_81 = permute_83 = getitem_165 = getitem_166 = getitem_171 = getitem_172 = None
        getitem_197: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_198: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_199: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_134: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_199, [0, 2, 1, 3]);  getitem_199 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_170: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_134, [4, 64, 768]);  permute_134 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_135: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_171: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_135, [4, 64, 768]);  permute_135 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_136: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_172: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_136, [4, 64, 768]);  permute_136 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_1: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_171, view_172, view_170], 2);  view_171 = view_172 = view_170 = None
        view_173: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_1, [256, 2304]);  cat_1 = None
        permute_137: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_65: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_137, view_120);  permute_137 = view_120 = None
        mm_66: "bf16[256, 768]" = torch.ops.aten.mm.default(view_173, permute_139);  view_173 = permute_139 = None
        view_174: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_66, [4, 64, 768]);  mm_66 = None
        convert_element_type_262: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        convert_element_type_263: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_131: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_262, primals_64);  primals_64 = None
        mul_132: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_131, 768)
        sum_17: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True)
        mul_133: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_131, mul_70);  mul_131 = None
        sum_18: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True);  mul_133 = None
        mul_134: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_18);  sum_18 = None
        sub_41: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_132, sum_17);  mul_132 = sum_17 = None
        sub_42: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_134);  sub_41 = mul_134 = None
        mul_135: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_42);  div_6 = sub_42 = None
        mul_136: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_262, mul_70);  convert_element_type_262 = mul_70 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_136, [0, 1]);  mul_136 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_70: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_69, mul_135);  add_69 = mul_135 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_264: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_175: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_264, [256, 768]);  convert_element_type_264 = None
        permute_141: "bf16[768, 256]" = torch.ops.aten.permute.default(view_175, [1, 0])
        mm_67: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_141, view_118);  permute_141 = view_118 = None
        mm_68: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_175, permute_143);  view_175 = permute_143 = None
        view_176: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_68, [4, 64, 3072]);  mm_68 = None
        convert_element_type_269: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_270: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_176, torch.float32);  view_176 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_117: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_38, [4, 64, 3072]);  mm_38 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_155: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_68: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476)
        erf_9: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_49: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_138: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_139: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_155)
        mul_140: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_139, -0.5);  mul_139 = None
        exp_4: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_140);  mul_140 = None
        mul_141: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_142: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, mul_141);  convert_element_type_155 = mul_141 = None
        add_72: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_138, mul_142);  mul_138 = mul_142 = None
        mul_143: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_270, add_72);  convert_element_type_270 = add_72 = None
        convert_element_type_272: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_177: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_272, [256, 3072]);  convert_element_type_272 = None
        permute_145: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_69: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_145, view_116);  permute_145 = view_116 = None
        mm_70: "bf16[256, 768]" = torch.ops.aten.mm.default(view_177, permute_147);  view_177 = permute_147 = None
        view_178: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_70, [4, 64, 768]);  mm_70 = None
        convert_element_type_277: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        convert_element_type_278: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_145: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_277, primals_61);  primals_61 = None
        mul_146: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_145, 768)
        sum_20: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_145, mul_65);  mul_145 = None
        sum_21: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_21);  sum_21 = None
        sub_44: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_146, sum_20);  mul_146 = sum_20 = None
        sub_45: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_148);  sub_44 = mul_148 = None
        mul_149: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_45);  div_7 = sub_45 = None
        mul_150: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_277, mul_65);  convert_element_type_277 = mul_65 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_73: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_70, mul_149);  add_70 = mul_149 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_279: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_179: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_279, [256, 768]);  convert_element_type_279 = None
        permute_149: "bf16[768, 256]" = torch.ops.aten.permute.default(view_179, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_76: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_113: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_76, [4, 64, 768]);  permute_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_114: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_113, [256, 768]);  view_113 = None
        mm_71: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_149, view_114);  permute_149 = view_114 = None
        mm_72: "bf16[256, 768]" = torch.ops.aten.mm.default(view_179, permute_151);  view_179 = permute_151 = None
        view_180: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_72, [4, 64, 768]);  mm_72 = None
        convert_element_type_284: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_181: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_180, [4, 64, 12, 64]);  view_180 = None
        permute_153: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_153, permute_74, permute_73, permute_75, getitem_149, getitem_150, None, None, 64, 64, 0.0, True, getitem_155, getitem_156, scale = 0.125);  permute_153 = permute_74 = permute_73 = permute_75 = getitem_149 = getitem_150 = getitem_155 = getitem_156 = None
        getitem_200: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_201: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_202: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_154: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_182: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_154, [4, 64, 768]);  permute_154 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_155: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_183: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_155, [4, 64, 768]);  permute_155 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_156: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_184: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_156, [4, 64, 768]);  permute_156 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_2: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_183, view_184, view_182], 2);  view_183 = view_184 = view_182 = None
        view_185: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_2, [256, 2304]);  cat_2 = None
        permute_157: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_73: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_157, view_108);  permute_157 = view_108 = None
        mm_74: "bf16[256, 768]" = torch.ops.aten.mm.default(view_185, permute_159);  view_185 = permute_159 = None
        view_186: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_74, [4, 64, 768]);  mm_74 = None
        convert_element_type_289: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        convert_element_type_290: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_152: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_289, primals_58);  primals_58 = None
        mul_153: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_23: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_152, mul_63);  mul_152 = None
        sum_24: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_24);  sum_24 = None
        sub_47: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_153, sum_23);  mul_153 = sum_23 = None
        sub_48: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_155);  sub_47 = mul_155 = None
        mul_156: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_48);  div_8 = sub_48 = None
        mul_157: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_289, mul_63);  convert_element_type_289 = mul_63 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_74: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_73, mul_156);  add_73 = mul_156 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_291: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_187: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_291, [256, 768]);  convert_element_type_291 = None
        permute_161: "bf16[768, 256]" = torch.ops.aten.permute.default(view_187, [1, 0])
        mm_75: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_161, view_106);  permute_161 = view_106 = None
        mm_76: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_187, permute_163);  view_187 = permute_163 = None
        view_188: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_76, [4, 64, 3072]);  mm_76 = None
        convert_element_type_296: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_297: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_105: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_34, [4, 64, 3072]);  mm_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_139: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_61: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476)
        erf_8: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_44: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_159: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_160: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, convert_element_type_139)
        mul_161: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
        exp_5: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_161);  mul_161 = None
        mul_162: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_163: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, mul_162);  convert_element_type_139 = mul_162 = None
        add_76: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
        mul_164: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_297, add_76);  convert_element_type_297 = add_76 = None
        convert_element_type_299: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_189: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_299, [256, 3072]);  convert_element_type_299 = None
        permute_165: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_77: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_165, view_104);  permute_165 = view_104 = None
        mm_78: "bf16[256, 768]" = torch.ops.aten.mm.default(view_189, permute_167);  view_189 = permute_167 = None
        view_190: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_78, [4, 64, 768]);  mm_78 = None
        convert_element_type_304: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        convert_element_type_305: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_166: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_304, primals_55);  primals_55 = None
        mul_167: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_166, 768)
        sum_26: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
        mul_168: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_166, mul_58);  mul_166 = None
        sum_27: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
        mul_169: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_27);  sum_27 = None
        sub_50: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_167, sum_26);  mul_167 = sum_26 = None
        sub_51: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_169);  sub_50 = mul_169 = None
        mul_170: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_51);  div_9 = sub_51 = None
        mul_171: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_304, mul_58);  convert_element_type_304 = mul_58 = None
        sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_77: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_74, mul_170);  add_74 = mul_170 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_306: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_191: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_306, [256, 768]);  convert_element_type_306 = None
        permute_169: "bf16[768, 256]" = torch.ops.aten.permute.default(view_191, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_68: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_101: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_68, [4, 64, 768]);  permute_68 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_102: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_101, [256, 768]);  view_101 = None
        mm_79: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_169, view_102);  permute_169 = view_102 = None
        mm_80: "bf16[256, 768]" = torch.ops.aten.mm.default(view_191, permute_171);  view_191 = permute_171 = None
        view_192: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_80, [4, 64, 768]);  mm_80 = None
        convert_element_type_311: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_193: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_192, [4, 64, 12, 64]);  view_192 = None
        permute_173: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_173, permute_66, permute_65, permute_67, getitem_133, getitem_134, None, None, 64, 64, 0.0, True, getitem_139, getitem_140, scale = 0.125);  permute_173 = permute_66 = permute_65 = permute_67 = getitem_133 = getitem_134 = getitem_139 = getitem_140 = None
        getitem_203: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_204: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_205: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_174: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_194: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_174, [4, 64, 768]);  permute_174 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_175: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_195: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_175, [4, 64, 768]);  permute_175 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_176: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_196: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_176, [4, 64, 768]);  permute_176 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_3: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_195, view_196, view_194], 2);  view_195 = view_196 = view_194 = None
        view_197: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_3, [256, 2304]);  cat_3 = None
        permute_177: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_81: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_177, view_96);  permute_177 = view_96 = None
        mm_82: "bf16[256, 768]" = torch.ops.aten.mm.default(view_197, permute_179);  view_197 = permute_179 = None
        view_198: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_82, [4, 64, 768]);  mm_82 = None
        convert_element_type_316: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        convert_element_type_317: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_173: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_316, primals_52);  primals_52 = None
        mul_174: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_173, 768)
        sum_29: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
        mul_175: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_173, mul_56);  mul_173 = None
        sum_30: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        mul_176: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_30);  sum_30 = None
        sub_53: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_174, sum_29);  mul_174 = sum_29 = None
        sub_54: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_176);  sub_53 = mul_176 = None
        mul_177: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_54);  div_10 = sub_54 = None
        mul_178: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_316, mul_56);  convert_element_type_316 = mul_56 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_78: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_77, mul_177);  add_77 = mul_177 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_318: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_199: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_318, [256, 768]);  convert_element_type_318 = None
        permute_181: "bf16[768, 256]" = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_83: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_181, view_94);  permute_181 = view_94 = None
        mm_84: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_199, permute_183);  view_199 = permute_183 = None
        view_200: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_84, [4, 64, 3072]);  mm_84 = None
        convert_element_type_323: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_324: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_200, torch.float32);  view_200 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_93: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_30, [4, 64, 3072]);  mm_30 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_123: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_54: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476)
        erf_7: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_39: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_180: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_181: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, convert_element_type_123)
        mul_182: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_181, -0.5);  mul_181 = None
        exp_6: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_182);  mul_182 = None
        mul_183: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_184: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, mul_183);  convert_element_type_123 = mul_183 = None
        add_80: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_180, mul_184);  mul_180 = mul_184 = None
        mul_185: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_324, add_80);  convert_element_type_324 = add_80 = None
        convert_element_type_326: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_201: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_326, [256, 3072]);  convert_element_type_326 = None
        permute_185: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_85: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_185, view_92);  permute_185 = view_92 = None
        mm_86: "bf16[256, 768]" = torch.ops.aten.mm.default(view_201, permute_187);  view_201 = permute_187 = None
        view_202: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_86, [4, 64, 768]);  mm_86 = None
        convert_element_type_331: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_202, torch.float32);  view_202 = None
        convert_element_type_332: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_187: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_331, primals_49);  primals_49 = None
        mul_188: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_187, 768)
        sum_32: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True)
        mul_189: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_187, mul_51);  mul_187 = None
        sum_33: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True);  mul_189 = None
        mul_190: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_33);  sum_33 = None
        sub_56: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_188, sum_32);  mul_188 = sum_32 = None
        sub_57: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_190);  sub_56 = mul_190 = None
        mul_191: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_57);  div_11 = sub_57 = None
        mul_192: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_331, mul_51);  convert_element_type_331 = mul_51 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1]);  mul_192 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_81: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_78, mul_191);  add_78 = mul_191 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_333: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_203: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_333, [256, 768]);  convert_element_type_333 = None
        permute_189: "bf16[768, 256]" = torch.ops.aten.permute.default(view_203, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_60: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_89: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_60, [4, 64, 768]);  permute_60 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_90: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_89, [256, 768]);  view_89 = None
        mm_87: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_189, view_90);  permute_189 = view_90 = None
        mm_88: "bf16[256, 768]" = torch.ops.aten.mm.default(view_203, permute_191);  view_203 = permute_191 = None
        view_204: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_88, [4, 64, 768]);  mm_88 = None
        convert_element_type_338: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_205: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_204, [4, 64, 12, 64]);  view_204 = None
        permute_193: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_193, permute_58, permute_57, permute_59, getitem_117, getitem_118, None, None, 64, 64, 0.0, True, getitem_123, getitem_124, scale = 0.125);  permute_193 = permute_58 = permute_57 = permute_59 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_206: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_207: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_208: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_194: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_206: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_194, [4, 64, 768]);  permute_194 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_195: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_206, [0, 2, 1, 3]);  getitem_206 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_207: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_195, [4, 64, 768]);  permute_195 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_196: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_208: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_196, [4, 64, 768]);  permute_196 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_4: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_207, view_208, view_206], 2);  view_207 = view_208 = view_206 = None
        view_209: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_4, [256, 2304]);  cat_4 = None
        permute_197: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_209, [1, 0])
        mm_89: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_197, view_84);  permute_197 = view_84 = None
        mm_90: "bf16[256, 768]" = torch.ops.aten.mm.default(view_209, permute_199);  view_209 = permute_199 = None
        view_210: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_90, [4, 64, 768]);  mm_90 = None
        convert_element_type_343: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_210, torch.float32);  view_210 = None
        convert_element_type_344: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_194: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_343, primals_46);  primals_46 = None
        mul_195: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_194, 768)
        sum_35: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True)
        mul_196: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_194, mul_49);  mul_194 = None
        sum_36: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True);  mul_196 = None
        mul_197: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_36);  sum_36 = None
        sub_59: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_195, sum_35);  mul_195 = sum_35 = None
        sub_60: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_197);  sub_59 = mul_197 = None
        mul_198: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_60);  div_12 = sub_60 = None
        mul_199: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_343, mul_49);  convert_element_type_343 = mul_49 = None
        sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_82: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_81, mul_198);  add_81 = mul_198 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_345: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_211: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_345, [256, 768]);  convert_element_type_345 = None
        permute_201: "bf16[768, 256]" = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_91: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_201, view_82);  permute_201 = view_82 = None
        mm_92: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_211, permute_203);  view_211 = permute_203 = None
        view_212: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_92, [4, 64, 3072]);  mm_92 = None
        convert_element_type_350: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_351: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_81: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_26, [4, 64, 3072]);  mm_26 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_107: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_47: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.7071067811865476)
        erf_6: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_34: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_201: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_202: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, convert_element_type_107)
        mul_203: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_202, -0.5);  mul_202 = None
        exp_7: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_203);  mul_203 = None
        mul_204: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_205: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, mul_204);  convert_element_type_107 = mul_204 = None
        add_84: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_201, mul_205);  mul_201 = mul_205 = None
        mul_206: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_351, add_84);  convert_element_type_351 = add_84 = None
        convert_element_type_353: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_213: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_353, [256, 3072]);  convert_element_type_353 = None
        permute_205: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_93: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_205, view_80);  permute_205 = view_80 = None
        mm_94: "bf16[256, 768]" = torch.ops.aten.mm.default(view_213, permute_207);  view_213 = permute_207 = None
        view_214: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_94, [4, 64, 768]);  mm_94 = None
        convert_element_type_358: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        convert_element_type_359: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_208: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_358, primals_43);  primals_43 = None
        mul_209: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
        sum_38: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
        mul_210: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_44);  mul_208 = None
        sum_39: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
        mul_211: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_39);  sum_39 = None
        sub_62: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_38);  mul_209 = sum_38 = None
        sub_63: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_211);  sub_62 = mul_211 = None
        mul_212: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_63);  div_13 = sub_63 = None
        mul_213: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_358, mul_44);  convert_element_type_358 = mul_44 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_85: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_82, mul_212);  add_82 = mul_212 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_360: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_215: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_360, [256, 768]);  convert_element_type_360 = None
        permute_209: "bf16[768, 256]" = torch.ops.aten.permute.default(view_215, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_52: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_77: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_52, [4, 64, 768]);  permute_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_78: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_77, [256, 768]);  view_77 = None
        mm_95: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_209, view_78);  permute_209 = view_78 = None
        mm_96: "bf16[256, 768]" = torch.ops.aten.mm.default(view_215, permute_211);  view_215 = permute_211 = None
        view_216: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_96, [4, 64, 768]);  mm_96 = None
        convert_element_type_365: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_217: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_216, [4, 64, 12, 64]);  view_216 = None
        permute_213: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_213, permute_50, permute_49, permute_51, getitem_101, getitem_102, None, None, 64, 64, 0.0, True, getitem_107, getitem_108, scale = 0.125);  permute_213 = permute_50 = permute_49 = permute_51 = getitem_101 = getitem_102 = getitem_107 = getitem_108 = None
        getitem_209: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_5[0]
        getitem_210: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_5[1]
        getitem_211: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_214: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_218: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_214, [4, 64, 768]);  permute_214 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_215: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_219: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_215, [4, 64, 768]);  permute_215 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_216: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_220: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_216, [4, 64, 768]);  permute_216 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_5: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_219, view_220, view_218], 2);  view_219 = view_220 = view_218 = None
        view_221: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_5, [256, 2304]);  cat_5 = None
        permute_217: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_97: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_217, view_72);  permute_217 = view_72 = None
        mm_98: "bf16[256, 768]" = torch.ops.aten.mm.default(view_221, permute_219);  view_221 = permute_219 = None
        view_222: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_98, [4, 64, 768]);  mm_98 = None
        convert_element_type_370: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_222, torch.float32);  view_222 = None
        convert_element_type_371: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_215: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_370, primals_40);  primals_40 = None
        mul_216: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_215, 768)
        sum_41: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
        mul_217: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_215, mul_42);  mul_215 = None
        sum_42: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
        mul_218: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_42);  sum_42 = None
        sub_65: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_216, sum_41);  mul_216 = sum_41 = None
        sub_66: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_218);  sub_65 = mul_218 = None
        mul_219: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_66);  div_14 = sub_66 = None
        mul_220: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_42);  convert_element_type_370 = mul_42 = None
        sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_86: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_85, mul_219);  add_85 = mul_219 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_372: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_223: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_372, [256, 768]);  convert_element_type_372 = None
        permute_221: "bf16[768, 256]" = torch.ops.aten.permute.default(view_223, [1, 0])
        mm_99: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_221, view_70);  permute_221 = view_70 = None
        mm_100: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_223, permute_223);  view_223 = permute_223 = None
        view_224: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_100, [4, 64, 3072]);  mm_100 = None
        convert_element_type_377: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_378: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_69: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_22, [4, 64, 3072]);  mm_22 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_91: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_40: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476)
        erf_5: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_29: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_222: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
        mul_223: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, convert_element_type_91)
        mul_224: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_223, -0.5);  mul_223 = None
        exp_8: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_224);  mul_224 = None
        mul_225: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_226: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, mul_225);  convert_element_type_91 = mul_225 = None
        add_88: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_222, mul_226);  mul_222 = mul_226 = None
        mul_227: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_378, add_88);  convert_element_type_378 = add_88 = None
        convert_element_type_380: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_225: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_380, [256, 3072]);  convert_element_type_380 = None
        permute_225: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_101: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_225, view_68);  permute_225 = view_68 = None
        mm_102: "bf16[256, 768]" = torch.ops.aten.mm.default(view_225, permute_227);  view_225 = permute_227 = None
        view_226: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_102, [4, 64, 768]);  mm_102 = None
        convert_element_type_385: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_226, torch.float32);  view_226 = None
        convert_element_type_386: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_229: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_385, primals_37);  primals_37 = None
        mul_230: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_229, 768)
        sum_44: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        mul_231: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_229, mul_37);  mul_229 = None
        sum_45: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_45);  sum_45 = None
        sub_68: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_230, sum_44);  mul_230 = sum_44 = None
        sub_69: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_232);  sub_68 = mul_232 = None
        mul_233: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_69);  div_15 = sub_69 = None
        mul_234: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_385, mul_37);  convert_element_type_385 = mul_37 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_89: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_86, mul_233);  add_86 = mul_233 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_387: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_227: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_387, [256, 768]);  convert_element_type_387 = None
        permute_229: "bf16[768, 256]" = torch.ops.aten.permute.default(view_227, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_44: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_65: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_44, [4, 64, 768]);  permute_44 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_66: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_65, [256, 768]);  view_65 = None
        mm_103: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_229, view_66);  permute_229 = view_66 = None
        mm_104: "bf16[256, 768]" = torch.ops.aten.mm.default(view_227, permute_231);  view_227 = permute_231 = None
        view_228: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_104, [4, 64, 768]);  mm_104 = None
        convert_element_type_392: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_229: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_228, [4, 64, 12, 64]);  view_228 = None
        permute_233: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_233, permute_42, permute_41, permute_43, getitem_85, getitem_86, None, None, 64, 64, 0.0, True, getitem_91, getitem_92, scale = 0.125);  permute_233 = permute_42 = permute_41 = permute_43 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_212: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_6[0]
        getitem_213: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_6[1]
        getitem_214: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_234: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_214, [0, 2, 1, 3]);  getitem_214 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_230: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_234, [4, 64, 768]);  permute_234 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_235: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3]);  getitem_212 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_231: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_235, [4, 64, 768]);  permute_235 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_236: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_213, [0, 2, 1, 3]);  getitem_213 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_232: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_236, [4, 64, 768]);  permute_236 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_6: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_231, view_232, view_230], 2);  view_231 = view_232 = view_230 = None
        view_233: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_6, [256, 2304]);  cat_6 = None
        permute_237: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_233, [1, 0])
        mm_105: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_237, view_60);  permute_237 = view_60 = None
        mm_106: "bf16[256, 768]" = torch.ops.aten.mm.default(view_233, permute_239);  view_233 = permute_239 = None
        view_234: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_106, [4, 64, 768]);  mm_106 = None
        convert_element_type_397: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        convert_element_type_398: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_236: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_397, primals_34);  primals_34 = None
        mul_237: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_47: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_35);  mul_236 = None
        sum_48: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_48);  sum_48 = None
        sub_71: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_47);  mul_237 = sum_47 = None
        sub_72: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_239);  sub_71 = mul_239 = None
        mul_240: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_72);  div_16 = sub_72 = None
        mul_241: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_397, mul_35);  convert_element_type_397 = mul_35 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_90: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_89, mul_240);  add_89 = mul_240 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_399: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_235: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_399, [256, 768]);  convert_element_type_399 = None
        permute_241: "bf16[768, 256]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_107: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_58);  permute_241 = view_58 = None
        mm_108: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_235, permute_243);  view_235 = permute_243 = None
        view_236: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_108, [4, 64, 3072]);  mm_108 = None
        convert_element_type_404: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_405: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_236, torch.float32);  view_236 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_57: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_18, [4, 64, 3072]);  mm_18 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_75: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_33: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476)
        erf_4: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_24: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_243: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_244: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, convert_element_type_75)
        mul_245: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_244, -0.5);  mul_244 = None
        exp_9: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_245);  mul_245 = None
        mul_246: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_247: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, mul_246);  convert_element_type_75 = mul_246 = None
        add_92: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_243, mul_247);  mul_243 = mul_247 = None
        mul_248: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_405, add_92);  convert_element_type_405 = add_92 = None
        convert_element_type_407: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_237: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_407, [256, 3072]);  convert_element_type_407 = None
        permute_245: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_109: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_56);  permute_245 = view_56 = None
        mm_110: "bf16[256, 768]" = torch.ops.aten.mm.default(view_237, permute_247);  view_237 = permute_247 = None
        view_238: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_110, [4, 64, 768]);  mm_110 = None
        convert_element_type_412: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        convert_element_type_413: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_250: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_412, primals_31);  primals_31 = None
        mul_251: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_250, 768)
        sum_50: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
        mul_252: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_250, mul_30);  mul_250 = None
        sum_51: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        mul_253: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_51);  sum_51 = None
        sub_74: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_251, sum_50);  mul_251 = sum_50 = None
        sub_75: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_253);  sub_74 = mul_253 = None
        mul_254: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_75);  div_17 = sub_75 = None
        mul_255: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_412, mul_30);  convert_element_type_412 = mul_30 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_93: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_90, mul_254);  add_90 = mul_254 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_414: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_239: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_414, [256, 768]);  convert_element_type_414 = None
        permute_249: "bf16[768, 256]" = torch.ops.aten.permute.default(view_239, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_36: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_53: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_36, [4, 64, 768]);  permute_36 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_54: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_53, [256, 768]);  view_53 = None
        mm_111: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_249, view_54);  permute_249 = view_54 = None
        mm_112: "bf16[256, 768]" = torch.ops.aten.mm.default(view_239, permute_251);  view_239 = permute_251 = None
        view_240: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_112, [4, 64, 768]);  mm_112 = None
        convert_element_type_419: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_241: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_240, [4, 64, 12, 64]);  view_240 = None
        permute_253: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_253, permute_34, permute_33, permute_35, getitem_69, getitem_70, None, None, 64, 64, 0.0, True, getitem_75, getitem_76, scale = 0.125);  permute_253 = permute_34 = permute_33 = permute_35 = getitem_69 = getitem_70 = getitem_75 = getitem_76 = None
        getitem_215: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_7[0]
        getitem_216: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_7[1]
        getitem_217: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_254: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_242: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_254, [4, 64, 768]);  permute_254 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_255: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_215, [0, 2, 1, 3]);  getitem_215 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_243: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_255, [4, 64, 768]);  permute_255 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_256: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_244: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_256, [4, 64, 768]);  permute_256 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_7: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_243, view_244, view_242], 2);  view_243 = view_244 = view_242 = None
        view_245: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_7, [256, 2304]);  cat_7 = None
        permute_257: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_245, [1, 0])
        mm_113: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_257, view_48);  permute_257 = view_48 = None
        mm_114: "bf16[256, 768]" = torch.ops.aten.mm.default(view_245, permute_259);  view_245 = permute_259 = None
        view_246: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_114, [4, 64, 768]);  mm_114 = None
        convert_element_type_424: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_425: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_257: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_424, primals_28);  primals_28 = None
        mul_258: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_257, 768)
        sum_53: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
        mul_259: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_28);  mul_257 = None
        sum_54: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
        mul_260: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_54);  sum_54 = None
        sub_77: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_258, sum_53);  mul_258 = sum_53 = None
        sub_78: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_260);  sub_77 = mul_260 = None
        mul_261: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_78);  div_18 = sub_78 = None
        mul_262: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_424, mul_28);  convert_element_type_424 = mul_28 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_94: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_93, mul_261);  add_93 = mul_261 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_426: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_247: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_426, [256, 768]);  convert_element_type_426 = None
        permute_261: "bf16[768, 256]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_115: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_261, view_46);  permute_261 = view_46 = None
        mm_116: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_247, permute_263);  view_247 = permute_263 = None
        view_248: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_116, [4, 64, 3072]);  mm_116 = None
        convert_element_type_431: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_432: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_45: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_14, [4, 64, 3072]);  mm_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_59: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_26: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476)
        erf_3: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_19: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_264: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_19, 0.5);  add_19 = None
        mul_265: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, convert_element_type_59)
        mul_266: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
        exp_10: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
        mul_267: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_268: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_267);  convert_element_type_59 = mul_267 = None
        add_96: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
        mul_269: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_432, add_96);  convert_element_type_432 = add_96 = None
        convert_element_type_434: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_249: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_434, [256, 3072]);  convert_element_type_434 = None
        permute_265: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_117: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_265, view_44);  permute_265 = view_44 = None
        mm_118: "bf16[256, 768]" = torch.ops.aten.mm.default(view_249, permute_267);  view_249 = permute_267 = None
        view_250: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_118, [4, 64, 768]);  mm_118 = None
        convert_element_type_439: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        convert_element_type_440: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_271: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_439, primals_25);  primals_25 = None
        mul_272: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
        sum_56: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
        mul_273: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_23);  mul_271 = None
        sum_57: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
        mul_274: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_57);  sum_57 = None
        sub_80: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_56);  mul_272 = sum_56 = None
        sub_81: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_274);  sub_80 = mul_274 = None
        mul_275: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_81);  div_19 = sub_81 = None
        mul_276: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_439, mul_23);  convert_element_type_439 = mul_23 = None
        sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_97: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_94, mul_275);  add_94 = mul_275 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_441: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_251: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_441, [256, 768]);  convert_element_type_441 = None
        permute_269: "bf16[768, 256]" = torch.ops.aten.permute.default(view_251, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_28: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_41: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_28, [4, 64, 768]);  permute_28 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_42: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_41, [256, 768]);  view_41 = None
        mm_119: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_269, view_42);  permute_269 = view_42 = None
        mm_120: "bf16[256, 768]" = torch.ops.aten.mm.default(view_251, permute_271);  view_251 = permute_271 = None
        view_252: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_120, [4, 64, 768]);  mm_120 = None
        convert_element_type_446: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_253: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_252, [4, 64, 12, 64]);  view_252 = None
        permute_273: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_273, permute_26, permute_25, permute_27, getitem_53, getitem_54, None, None, 64, 64, 0.0, True, getitem_59, getitem_60, scale = 0.125);  permute_273 = permute_26 = permute_25 = permute_27 = getitem_53 = getitem_54 = getitem_59 = getitem_60 = None
        getitem_218: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_8[0]
        getitem_219: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_8[1]
        getitem_220: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_274: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_220, [0, 2, 1, 3]);  getitem_220 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_254: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_274, [4, 64, 768]);  permute_274 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_275: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_255: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_275, [4, 64, 768]);  permute_275 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_276: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_219, [0, 2, 1, 3]);  getitem_219 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_256: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_276, [4, 64, 768]);  permute_276 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_8: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_255, view_256, view_254], 2);  view_255 = view_256 = view_254 = None
        view_257: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_8, [256, 2304]);  cat_8 = None
        permute_277: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_121: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_277, view_36);  permute_277 = view_36 = None
        mm_122: "bf16[256, 768]" = torch.ops.aten.mm.default(view_257, permute_279);  view_257 = permute_279 = None
        view_258: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_122, [4, 64, 768]);  mm_122 = None
        convert_element_type_451: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        convert_element_type_452: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_278: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_451, primals_22);  primals_22 = None
        mul_279: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_278, 768)
        sum_59: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_278, mul_21);  mul_278 = None
        sum_60: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_60);  sum_60 = None
        sub_83: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_279, sum_59);  mul_279 = sum_59 = None
        sub_84: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_281);  sub_83 = mul_281 = None
        mul_282: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_84);  div_20 = sub_84 = None
        mul_283: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_451, mul_21);  convert_element_type_451 = mul_21 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_98: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_97, mul_282);  add_97 = mul_282 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_453: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_259: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_453, [256, 768]);  convert_element_type_453 = None
        permute_281: "bf16[768, 256]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_123: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_281, view_34);  permute_281 = view_34 = None
        mm_124: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_259, permute_283);  view_259 = permute_283 = None
        view_260: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_124, [4, 64, 3072]);  mm_124 = None
        convert_element_type_458: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_459: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_33: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_10, [4, 64, 3072]);  mm_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_43: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_19: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476)
        erf_2: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_285: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_286: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, convert_element_type_43)
        mul_287: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
        exp_11: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
        mul_288: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_289: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, mul_288);  convert_element_type_43 = mul_288 = None
        add_100: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
        mul_290: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_459, add_100);  convert_element_type_459 = add_100 = None
        convert_element_type_461: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_261: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_461, [256, 3072]);  convert_element_type_461 = None
        permute_285: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_125: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_285, view_32);  permute_285 = view_32 = None
        mm_126: "bf16[256, 768]" = torch.ops.aten.mm.default(view_261, permute_287);  view_261 = permute_287 = None
        view_262: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_126, [4, 64, 768]);  mm_126 = None
        convert_element_type_466: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        convert_element_type_467: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_292: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_466, primals_19);  primals_19 = None
        mul_293: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_292, 768)
        sum_62: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
        mul_294: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_292, mul_16);  mul_292 = None
        sum_63: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
        mul_295: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_63);  sum_63 = None
        sub_86: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_293, sum_62);  mul_293 = sum_62 = None
        sub_87: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_295);  sub_86 = mul_295 = None
        mul_296: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_87);  div_21 = sub_87 = None
        mul_297: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_466, mul_16);  convert_element_type_466 = mul_16 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_101: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_98, mul_296);  add_98 = mul_296 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_468: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_263: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_468, [256, 768]);  convert_element_type_468 = None
        permute_289: "bf16[768, 256]" = torch.ops.aten.permute.default(view_263, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_20: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_29: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_20, [4, 64, 768]);  permute_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_30: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_29, [256, 768]);  view_29 = None
        mm_127: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_289, view_30);  permute_289 = view_30 = None
        mm_128: "bf16[256, 768]" = torch.ops.aten.mm.default(view_263, permute_291);  view_263 = permute_291 = None
        view_264: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_128, [4, 64, 768]);  mm_128 = None
        convert_element_type_473: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_265: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_264, [4, 64, 12, 64]);  view_264 = None
        permute_293: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_293, permute_18, permute_17, permute_19, getitem_37, getitem_38, None, None, 64, 64, 0.0, True, getitem_43, getitem_44, scale = 0.125);  permute_293 = permute_18 = permute_17 = permute_19 = getitem_37 = getitem_38 = getitem_43 = getitem_44 = None
        getitem_221: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_9[0]
        getitem_222: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_9[1]
        getitem_223: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_294: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_266: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_294, [4, 64, 768]);  permute_294 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_295: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_221, [0, 2, 1, 3]);  getitem_221 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_267: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_295, [4, 64, 768]);  permute_295 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_296: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3]);  getitem_222 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_268: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_296, [4, 64, 768]);  permute_296 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_9: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_267, view_268, view_266], 2);  view_267 = view_268 = view_266 = None
        view_269: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_9, [256, 2304]);  cat_9 = None
        permute_297: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_129: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_297, view_24);  permute_297 = view_24 = None
        mm_130: "bf16[256, 768]" = torch.ops.aten.mm.default(view_269, permute_299);  view_269 = permute_299 = None
        view_270: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_130, [4, 64, 768]);  mm_130 = None
        convert_element_type_478: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        convert_element_type_479: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_299: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_478, primals_16);  primals_16 = None
        mul_300: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_299, 768)
        sum_65: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_299, mul_14);  mul_299 = None
        sum_66: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_66);  sum_66 = None
        sub_89: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_300, sum_65);  mul_300 = sum_65 = None
        sub_90: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_302);  sub_89 = mul_302 = None
        mul_303: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_90);  div_22 = sub_90 = None
        mul_304: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_478, mul_14);  convert_element_type_478 = mul_14 = None
        sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_102: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_101, mul_303);  add_101 = mul_303 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_480: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_271: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_480, [256, 768]);  convert_element_type_480 = None
        permute_301: "bf16[768, 256]" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_131: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_301, view_22);  permute_301 = view_22 = None
        mm_132: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_271, permute_303);  view_271 = permute_303 = None
        view_272: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_132, [4, 64, 3072]);  mm_132 = None
        convert_element_type_485: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_486: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_21: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_6, [4, 64, 3072]);  mm_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_27: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_12: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.7071067811865476)
        erf_1: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_306: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_307: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, convert_element_type_27)
        mul_308: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_307, -0.5);  mul_307 = None
        exp_12: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_308);  mul_308 = None
        mul_309: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_310: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, mul_309);  convert_element_type_27 = mul_309 = None
        add_104: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_306, mul_310);  mul_306 = mul_310 = None
        mul_311: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_486, add_104);  convert_element_type_486 = add_104 = None
        convert_element_type_488: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_273: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_488, [256, 3072]);  convert_element_type_488 = None
        permute_305: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_133: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_305, view_20);  permute_305 = view_20 = None
        mm_134: "bf16[256, 768]" = torch.ops.aten.mm.default(view_273, permute_307);  view_273 = permute_307 = None
        view_274: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_134, [4, 64, 768]);  mm_134 = None
        convert_element_type_493: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        convert_element_type_494: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_313: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_493, primals_13);  primals_13 = None
        mul_314: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
        sum_68: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
        mul_315: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_9);  mul_313 = None
        sum_69: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
        mul_316: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_69);  sum_69 = None
        sub_92: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_68);  mul_314 = sum_68 = None
        sub_93: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_316);  sub_92 = mul_316 = None
        mul_317: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_93);  div_23 = sub_93 = None
        mul_318: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_493, mul_9);  convert_element_type_493 = mul_9 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_105: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_102, mul_317);  add_102 = mul_317 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_495: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_275: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_495, [256, 768]);  convert_element_type_495 = None
        permute_309: "bf16[768, 256]" = torch.ops.aten.permute.default(view_275, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_12: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_17: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_12, [4, 64, 768]);  permute_12 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_18: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_17, [256, 768]);  view_17 = None
        mm_135: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_309, view_18);  permute_309 = view_18 = None
        mm_136: "bf16[256, 768]" = torch.ops.aten.mm.default(view_275, permute_311);  view_275 = permute_311 = None
        view_276: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_136, [4, 64, 768]);  mm_136 = None
        convert_element_type_500: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_277: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_276, [4, 64, 12, 64]);  view_276 = None
        permute_313: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_313, permute_10, permute_9, permute_11, getitem_21, getitem_22, None, None, 64, 64, 0.0, True, getitem_27, getitem_28, scale = 0.125);  permute_313 = permute_10 = permute_9 = permute_11 = getitem_21 = getitem_22 = getitem_27 = getitem_28 = None
        getitem_224: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_10[0]
        getitem_225: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_10[1]
        getitem_226: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_314: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3]);  getitem_226 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_278: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_314, [4, 64, 768]);  permute_314 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_315: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_279: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_315, [4, 64, 768]);  permute_315 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_316: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_280: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_316, [4, 64, 768]);  permute_316 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_10: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_279, view_280, view_278], 2);  view_279 = view_280 = view_278 = None
        view_281: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_10, [256, 2304]);  cat_10 = None
        permute_317: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_137: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_317, view_12);  permute_317 = view_12 = None
        mm_138: "bf16[256, 768]" = torch.ops.aten.mm.default(view_281, permute_319);  view_281 = permute_319 = None
        view_282: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_138, [4, 64, 768]);  mm_138 = None
        convert_element_type_505: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        convert_element_type_506: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_320: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_505, primals_10);  primals_10 = None
        mul_321: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_320, 768)
        sum_71: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        mul_322: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_320, mul_7);  mul_320 = None
        sum_72: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_72);  sum_72 = None
        sub_95: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_321, sum_71);  mul_321 = sum_71 = None
        sub_96: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_323);  sub_95 = mul_323 = None
        mul_324: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_96);  div_24 = sub_96 = None
        mul_325: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_505, mul_7);  convert_element_type_505 = mul_7 = None
        sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_106: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_105, mul_324);  add_105 = mul_324 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        convert_element_type_507: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        view_283: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_507, [256, 768]);  convert_element_type_507 = None
        permute_321: "bf16[768, 256]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_139: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_321, view_10);  permute_321 = view_10 = None
        mm_140: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_283, permute_323);  view_283 = permute_323 = None
        view_284: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_140, [4, 64, 3072]);  mm_140 = None
        convert_element_type_512: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_513: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_9: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_2, [4, 64, 3072]);  mm_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_11: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_5: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476)
        erf: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_327: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_328: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, convert_element_type_11)
        mul_329: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_328, -0.5);  mul_328 = None
        exp_13: "f32[4, 64, 3072]" = torch.ops.aten.exp.default(mul_329);  mul_329 = None
        mul_330: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_331: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_330);  convert_element_type_11 = mul_330 = None
        add_108: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(mul_327, mul_331);  mul_327 = mul_331 = None
        mul_332: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_513, add_108);  convert_element_type_513 = add_108 = None
        convert_element_type_515: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        view_285: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_515, [256, 3072]);  convert_element_type_515 = None
        permute_325: "bf16[3072, 256]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_141: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_325, view_8);  permute_325 = view_8 = None
        mm_142: "bf16[256, 768]" = torch.ops.aten.mm.default(view_285, permute_327);  view_285 = permute_327 = None
        view_286: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_142, [4, 64, 768]);  mm_142 = None
        convert_element_type_520: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        convert_element_type_521: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_334: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_520, primals_7);  primals_7 = None
        mul_335: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_334, 768)
        sum_74: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True)
        mul_336: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_334, mul_2);  mul_334 = None
        sum_75: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True);  mul_336 = None
        mul_337: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_75);  sum_75 = None
        sub_98: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_335, sum_74);  mul_335 = sum_74 = None
        sub_99: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_337);  sub_98 = mul_337 = None
        mul_338: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_99);  div_25 = sub_99 = None
        mul_339: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_520, mul_2);  convert_element_type_520 = mul_2 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1]);  mul_339 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_109: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_106, mul_338);  add_106 = mul_338 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        convert_element_type_522: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_287: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_522, [256, 768]);  convert_element_type_522 = None
        permute_329: "bf16[768, 256]" = torch.ops.aten.permute.default(view_287, [1, 0])
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_4: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_5: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_4, [4, 64, 768]);  permute_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        view_6: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_5, [256, 768]);  view_5 = None
        mm_143: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_329, view_6);  permute_329 = view_6 = None
        mm_144: "bf16[256, 768]" = torch.ops.aten.mm.default(view_287, permute_331);  view_287 = permute_331 = None
        view_288: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_144, [4, 64, 768]);  mm_144 = None
        convert_element_type_527: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        view_289: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(view_288, [4, 64, 12, 64]);  view_288 = None
        permute_333: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_333, permute_2, permute_1, permute_3, getitem_5, getitem_6, None, None, 64, 64, 0.0, True, getitem_11, getitem_12, scale = 0.125);  permute_333 = permute_2 = permute_1 = permute_3 = getitem_5 = getitem_6 = getitem_11 = getitem_12 = None
        getitem_227: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_11[0]
        getitem_228: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_11[1]
        getitem_229: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_334: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_290: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_334, [4, 64, 768]);  permute_334 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_335: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_291: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_335, [4, 64, 768]);  permute_335 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_336: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_228, [0, 2, 1, 3]);  getitem_228 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_292: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_336, [4, 64, 768]);  permute_336 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        cat_11: "bf16[4, 64, 2304]" = torch.ops.aten.cat.default([view_291, view_292, view_290], 2);  view_291 = view_292 = view_290 = None
        view_293: "bf16[256, 2304]" = torch.ops.aten.reshape.default(cat_11, [256, 2304]);  cat_11 = None
        permute_337: "bf16[2304, 256]" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_145: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_337, view);  permute_337 = view = None
        mm_146: "bf16[256, 768]" = torch.ops.aten.mm.default(view_293, permute_339);  view_293 = permute_339 = None
        view_294: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_146, [4, 64, 768]);  mm_146 = None
        convert_element_type_532: "f32[4, 64, 768]" = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None
        convert_element_type_533: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        mul_341: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_532, primals_4);  primals_4 = None
        mul_342: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_77: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        
         # File: /home/ubuntu/nanoGPT/model.py:316 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        add: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        sub: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_343: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul);  mul_341 = None
        sum_78: "f32[4, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul, sum_78);  sum_78 = None
        sub_101: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_77);  mul_342 = sum_77 = None
        sub_102: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_344);  sub_101 = mul_344 = None
        div_26: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_345: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_102);  div_26 = sub_102 = None
        mul_346: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_532, mul);  convert_element_type_532 = mul = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        add_110: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_109, mul_345);  add_109 = mul_345 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:316 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        sum_80: "f32[1, 64, 768]" = torch.ops.aten.sum.dim_IntList(add_110, [0], True, dtype = torch.float32)
        view_295: "f32[64, 768]" = torch.ops.aten.reshape.default(sum_80, [64, 768]);  sum_80 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:315 in forward, code: pos_emb = self.transformer.wpe(pos)
        eq: "b8[64]" = torch.ops.aten.eq.Scalar(iota, -1)
        unsqueeze_2: "b8[64, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_4: "f32[64, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_1, view_295);  unsqueeze_2 = view_295 = None
        full_default_6: "f32[64, 768]" = torch.ops.aten.full.default([64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[64, 768]" = torch.ops.aten.index_put_.default(full_default_6, [iota], where_4, True);  full_default_6 = iota = where_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:312 in forward, code: tok_emb = self.transformer.wte(idx)
        eq_1: "b8[4, 64]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_3: "b8[4, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5: "f32[4, 64, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, add_110);  unsqueeze_3 = full_default_1 = add_110 = None
        full_default_8: "f32[65, 768]" = torch.ops.aten.full.default([65, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[65, 768]" = torch.ops.aten.index_put_.default(full_default_8, [primals_1], where_5, True);  full_default_8 = primals_1 = where_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:312 in forward, code: tok_emb = self.transformer.wte(idx)
        add_111: "f32[65, 768]" = torch.ops.aten.add.Tensor(convert_element_type_209, index_put_1);  convert_element_type_209 = index_put_1 = None
        return (None, add_111, index_put, sum_79, convert_element_type_533, convert_element_type_527, sum_76, convert_element_type_521, convert_element_type_512, sum_73, convert_element_type_506, convert_element_type_500, sum_70, convert_element_type_494, convert_element_type_485, sum_67, convert_element_type_479, convert_element_type_473, sum_64, convert_element_type_467, convert_element_type_458, sum_61, convert_element_type_452, convert_element_type_446, sum_58, convert_element_type_440, convert_element_type_431, sum_55, convert_element_type_425, convert_element_type_419, sum_52, convert_element_type_413, convert_element_type_404, sum_49, convert_element_type_398, convert_element_type_392, sum_46, convert_element_type_386, convert_element_type_377, sum_43, convert_element_type_371, convert_element_type_365, sum_40, convert_element_type_359, convert_element_type_350, sum_37, convert_element_type_344, convert_element_type_338, sum_34, convert_element_type_332, convert_element_type_323, sum_31, convert_element_type_317, convert_element_type_311, sum_28, convert_element_type_305, convert_element_type_296, sum_25, convert_element_type_290, convert_element_type_284, sum_22, convert_element_type_278, convert_element_type_269, sum_19, convert_element_type_263, convert_element_type_257, sum_16, convert_element_type_251, convert_element_type_242, sum_13, convert_element_type_236, convert_element_type_230, sum_10, convert_element_type_224, convert_element_type_215, sum_7, None)
        