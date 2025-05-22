class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 64]", primals_2: "f32[65, 768]", primals_3: "f32[64, 768]", primals_4: "f32[768]", primals_5: "f32[2304, 768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[3072, 768]", primals_9: "f32[768, 3072]", primals_10: "f32[768]", primals_11: "f32[2304, 768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[3072, 768]", primals_15: "f32[768, 3072]", primals_16: "f32[768]", primals_17: "f32[2304, 768]", primals_18: "f32[768, 768]", primals_19: "f32[768]", primals_20: "f32[3072, 768]", primals_21: "f32[768, 3072]", primals_22: "f32[768]", primals_23: "f32[2304, 768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[3072, 768]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[2304, 768]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[768, 3072]", primals_34: "f32[768]", primals_35: "f32[2304, 768]", primals_36: "f32[768, 768]", primals_37: "f32[768]", primals_38: "f32[3072, 768]", primals_39: "f32[768, 3072]", primals_40: "f32[768]", primals_41: "f32[2304, 768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[3072, 768]", primals_45: "f32[768, 3072]", primals_46: "f32[768]", primals_47: "f32[2304, 768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[3072, 768]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[2304, 768]", primals_54: "f32[768, 768]", primals_55: "f32[768]", primals_56: "f32[3072, 768]", primals_57: "f32[768, 3072]", primals_58: "f32[768]", primals_59: "f32[2304, 768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[3072, 768]", primals_63: "f32[768, 3072]", primals_64: "f32[768]", primals_65: "f32[2304, 768]", primals_66: "f32[768, 768]", primals_67: "f32[768]", primals_68: "f32[3072, 768]", primals_69: "f32[768, 3072]", primals_70: "f32[768]", primals_71: "f32[2304, 768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[3072, 768]", primals_75: "f32[768, 3072]", primals_76: "f32[768]", primals_77: "i64[4, 64]"):
         # File: /home/ubuntu/nanoGPT/model.py:306 in forward, code: pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        iota: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/ubuntu/nanoGPT/model.py:312 in forward, code: tok_emb = self.transformer.wte(idx)
        embedding: "f32[4, 64, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1)
        
         # File: /home/ubuntu/nanoGPT/model.py:315 in forward, code: pos_emb = self.transformer.wpe(pos)
        embedding_1: "f32[64, 768]" = torch.ops.aten.embedding.default(primals_3, iota);  primals_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:316 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        add: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1)
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 64, 1]" = var_mean[0]
        getitem_1: "f32[4, 64, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [256, 768]);  convert_element_type_1 = None
        mm: "bf16[256, 2304]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm, [4, 64, 2304]);  mm = None
        split = torch.ops.aten.split.Tensor(view_1, 768, 2);  view_1 = None
        getitem_2: "bf16[4, 64, 768]" = split[0]
        getitem_3: "bf16[4, 64, 768]" = split[1]
        getitem_4: "bf16[4, 64, 768]" = split[2];  split = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_2: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [4, 64, 12, 64]);  getitem_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_1: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_3: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [4, 64, 12, 64]);  getitem_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_2: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_4: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [4, 64, 12, 64]);  getitem_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_3: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_2, permute_1, permute_3, 0.0, True, scale = 0.125)
        getitem_5: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention[0]
        getitem_6: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention[1]
        getitem_11: "u64[2]" = _scaled_dot_product_flash_attention[6]
        getitem_12: "u64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_4: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_5: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_4, [4, 64, 768]);  permute_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_4: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_5: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        view_6: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_5, [256, 768]);  view_5 = None
        mm_1: "bf16[256, 768]" = torch.ops.aten.mm.default(view_6, permute_5);  view_6 = None
        view_7: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_1, [4, 64, 768]);  mm_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_2: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add, view_7);  add = view_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 64, 1]" = var_mean_1[0]
        getitem_15: "f32[4, 64, 1]" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_1: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_15);  getitem_15 = None
        mul_2: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_7)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_7: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convert_element_type_8: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_6: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        view_8: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_8, [256, 768]);  convert_element_type_8 = None
        mm_2: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_8, permute_6)
        view_9: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_2, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_11: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_4: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5)
        mul_5: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476);  convert_element_type_11 = None
        erf: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_12: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_13: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        permute_7: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        view_10: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_12, [256, 3072]);  convert_element_type_12 = None
        mm_3: "bf16[256, 768]" = torch.ops.aten.mm.default(view_10, permute_7)
        view_11: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_3, [4, 64, 768]);  mm_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_5: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 64, 1]" = var_mean_2[0]
        getitem_17: "f32[4, 64, 1]" = var_mean_2[1];  var_mean_2 = None
        add_6: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_2: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_17);  getitem_17 = None
        mul_7: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_8: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_10)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_16: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_17: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        permute_8: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        view_12: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_17, [256, 768]);  convert_element_type_17 = None
        mm_4: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_12, permute_8)
        view_13: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_4, [4, 64, 2304]);  mm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_13, 768, 2);  view_13 = None
        getitem_18: "bf16[4, 64, 768]" = split_1[0]
        getitem_19: "bf16[4, 64, 768]" = split_1[1]
        getitem_20: "bf16[4, 64, 768]" = split_1[2];  split_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_14: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_19, [4, 64, 12, 64]);  getitem_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_9: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_15: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_18, [4, 64, 12, 64]);  getitem_18 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_10: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_16: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_20, [4, 64, 12, 64]);  getitem_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_11: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_10, permute_9, permute_11, 0.0, True, scale = 0.125)
        getitem_21: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_1[0]
        getitem_22: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_1[1]
        getitem_27: "u64[2]" = _scaled_dot_product_flash_attention_1[6]
        getitem_28: "u64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_12: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_17: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_12, [4, 64, 768]);  permute_12 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_20: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        view_18: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_17, [256, 768]);  view_17 = None
        mm_5: "bf16[256, 768]" = torch.ops.aten.mm.default(view_18, permute_13);  view_18 = None
        view_19: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_5, [4, 64, 768]);  mm_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_7: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_5, view_19);  add_5 = view_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 64, 1]" = var_mean_3[0]
        getitem_31: "f32[4, 64, 1]" = var_mean_3[1];  var_mean_3 = None
        add_8: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_3: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_31);  getitem_31 = None
        mul_9: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_10: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_13)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_23: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_24: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        permute_14: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        view_20: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_24, [256, 768]);  convert_element_type_24 = None
        mm_6: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_20, permute_14)
        view_21: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_6, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_27: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_11: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.5)
        mul_12: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.7071067811865476);  convert_element_type_27 = None
        erf_1: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = add_9 = None
        convert_element_type_28: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_29: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        permute_15: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_22: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_28, [256, 3072]);  convert_element_type_28 = None
        mm_7: "bf16[256, 768]" = torch.ops.aten.mm.default(view_22, permute_15)
        view_23: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_7, [4, 64, 768]);  mm_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_10: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_7, view_23);  add_7 = view_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 64, 1]" = var_mean_4[0]
        getitem_33: "f32[4, 64, 1]" = var_mean_4[1];  var_mean_4 = None
        add_11: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_4: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_33);  getitem_33 = None
        mul_14: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_15: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_16)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_32: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_33: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        permute_16: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        view_24: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_33, [256, 768]);  convert_element_type_33 = None
        mm_8: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_24, permute_16)
        view_25: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_8, [4, 64, 2304]);  mm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_25, 768, 2);  view_25 = None
        getitem_34: "bf16[4, 64, 768]" = split_2[0]
        getitem_35: "bf16[4, 64, 768]" = split_2[1]
        getitem_36: "bf16[4, 64, 768]" = split_2[2];  split_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_26: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [4, 64, 12, 64]);  getitem_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_17: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_27: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_34, [4, 64, 12, 64]);  getitem_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_18: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_28: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [4, 64, 12, 64]);  getitem_36 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_19: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_18, permute_17, permute_19, 0.0, True, scale = 0.125)
        getitem_37: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_2[0]
        getitem_38: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_2[1]
        getitem_43: "u64[2]" = _scaled_dot_product_flash_attention_2[6]
        getitem_44: "u64[]" = _scaled_dot_product_flash_attention_2[7];  _scaled_dot_product_flash_attention_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_20: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_29: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_20, [4, 64, 768]);  permute_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_36: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        permute_21: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        view_30: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_29, [256, 768]);  view_29 = None
        mm_9: "bf16[256, 768]" = torch.ops.aten.mm.default(view_30, permute_21);  view_30 = None
        view_31: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_9, [4, 64, 768]);  mm_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_12: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_10, view_31);  add_10 = view_31 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 64, 1]" = var_mean_5[0]
        getitem_47: "f32[4, 64, 1]" = var_mean_5[1];  var_mean_5 = None
        add_13: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_5: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_47);  getitem_47 = None
        mul_16: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_17: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_19)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_39: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_40: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        permute_22: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        view_32: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_40, [256, 768]);  convert_element_type_40 = None
        mm_10: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_32, permute_22)
        view_33: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_10, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_43: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_18: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_19: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_2: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
        convert_element_type_44: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_45: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        permute_23: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        view_34: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_44, [256, 3072]);  convert_element_type_44 = None
        mm_11: "bf16[256, 768]" = torch.ops.aten.mm.default(view_34, permute_23)
        view_35: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_11, [4, 64, 768]);  mm_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_15: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_12, view_35);  add_12 = view_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 64, 1]" = var_mean_6[0]
        getitem_49: "f32[4, 64, 1]" = var_mean_6[1];  var_mean_6 = None
        add_16: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_6: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_49);  getitem_49 = None
        mul_21: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_22: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_22)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_48: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_49: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        permute_24: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        view_36: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_49, [256, 768]);  convert_element_type_49 = None
        mm_12: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_36, permute_24)
        view_37: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_12, [4, 64, 2304]);  mm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_37, 768, 2);  view_37 = None
        getitem_50: "bf16[4, 64, 768]" = split_3[0]
        getitem_51: "bf16[4, 64, 768]" = split_3[1]
        getitem_52: "bf16[4, 64, 768]" = split_3[2];  split_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_38: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_51, [4, 64, 12, 64]);  getitem_51 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_25: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_39: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_50, [4, 64, 12, 64]);  getitem_50 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_26: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_40: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_52, [4, 64, 12, 64]);  getitem_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_27: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_26, permute_25, permute_27, 0.0, True, scale = 0.125)
        getitem_53: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_3[0]
        getitem_54: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_3[1]
        getitem_59: "u64[2]" = _scaled_dot_product_flash_attention_3[6]
        getitem_60: "u64[]" = _scaled_dot_product_flash_attention_3[7];  _scaled_dot_product_flash_attention_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_28: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_41: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_28, [4, 64, 768]);  permute_28 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_52: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_29: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        view_42: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_41, [256, 768]);  view_41 = None
        mm_13: "bf16[256, 768]" = torch.ops.aten.mm.default(view_42, permute_29);  view_42 = None
        view_43: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_13, [4, 64, 768]);  mm_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_17: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_15, view_43);  add_15 = view_43 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 64, 1]" = var_mean_7[0]
        getitem_63: "f32[4, 64, 1]" = var_mean_7[1];  var_mean_7 = None
        add_18: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_7: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_63);  getitem_63 = None
        mul_23: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_24: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_25)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_55: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_56: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        permute_30: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        view_44: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_56, [256, 768]);  convert_element_type_56 = None
        mm_14: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_44, permute_30)
        view_45: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_14, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_59: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_25: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.5)
        mul_26: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476);  convert_element_type_59 = None
        erf_3: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_19: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_19);  mul_25 = add_19 = None
        convert_element_type_60: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_61: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_31: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        view_46: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_60, [256, 3072]);  convert_element_type_60 = None
        mm_15: "bf16[256, 768]" = torch.ops.aten.mm.default(view_46, permute_31)
        view_47: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_15, [4, 64, 768]);  mm_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_20: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_17, view_47);  add_17 = view_47 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 64, 1]" = var_mean_8[0]
        getitem_65: "f32[4, 64, 1]" = var_mean_8[1];  var_mean_8 = None
        add_21: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_8: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_8: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_65);  getitem_65 = None
        mul_28: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_29: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_28)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_64: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_65: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        permute_32: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_48: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_65, [256, 768]);  convert_element_type_65 = None
        mm_16: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_48, permute_32)
        view_49: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_16, [4, 64, 2304]);  mm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_49, 768, 2);  view_49 = None
        getitem_66: "bf16[4, 64, 768]" = split_4[0]
        getitem_67: "bf16[4, 64, 768]" = split_4[1]
        getitem_68: "bf16[4, 64, 768]" = split_4[2];  split_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_50: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_67, [4, 64, 12, 64]);  getitem_67 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_33: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_51: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_66, [4, 64, 12, 64]);  getitem_66 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_34: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_52: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [4, 64, 12, 64]);  getitem_68 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_35: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_34, permute_33, permute_35, 0.0, True, scale = 0.125)
        getitem_69: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_4[0]
        getitem_70: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_4[1]
        getitem_75: "u64[2]" = _scaled_dot_product_flash_attention_4[6]
        getitem_76: "u64[]" = _scaled_dot_product_flash_attention_4[7];  _scaled_dot_product_flash_attention_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_36: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_53: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_36, [4, 64, 768]);  permute_36 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_68: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_37: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_54: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_53, [256, 768]);  view_53 = None
        mm_17: "bf16[256, 768]" = torch.ops.aten.mm.default(view_54, permute_37);  view_54 = None
        view_55: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_17, [4, 64, 768]);  mm_17 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_22: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_20, view_55);  add_20 = view_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 64, 1]" = var_mean_9[0]
        getitem_79: "f32[4, 64, 1]" = var_mean_9[1];  var_mean_9 = None
        add_23: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_9: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_9: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_79);  getitem_79 = None
        mul_30: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_31: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_31)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_71: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_72: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        permute_38: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        view_56: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_72, [256, 768]);  convert_element_type_72 = None
        mm_18: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_56, permute_38)
        view_57: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_18, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_75: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_32: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.5)
        mul_33: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476);  convert_element_type_75 = None
        erf_4: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_24: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_24);  mul_32 = add_24 = None
        convert_element_type_76: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_77: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_39: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        view_58: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_76, [256, 3072]);  convert_element_type_76 = None
        mm_19: "bf16[256, 768]" = torch.ops.aten.mm.default(view_58, permute_39)
        view_59: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_19, [4, 64, 768]);  mm_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_25: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_22, view_59);  add_22 = view_59 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 64, 1]" = var_mean_10[0]
        getitem_81: "f32[4, 64, 1]" = var_mean_10[1];  var_mean_10 = None
        add_26: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_10: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_81);  getitem_81 = None
        mul_35: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_36: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_34)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_80: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_81: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        permute_40: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_60: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_81, [256, 768]);  convert_element_type_81 = None
        mm_20: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_60, permute_40)
        view_61: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_20, [4, 64, 2304]);  mm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_61, 768, 2);  view_61 = None
        getitem_82: "bf16[4, 64, 768]" = split_5[0]
        getitem_83: "bf16[4, 64, 768]" = split_5[1]
        getitem_84: "bf16[4, 64, 768]" = split_5[2];  split_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_62: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_83, [4, 64, 12, 64]);  getitem_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_41: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_63: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_82, [4, 64, 12, 64]);  getitem_82 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_42: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_64: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_84, [4, 64, 12, 64]);  getitem_84 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_43: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_42, permute_41, permute_43, 0.0, True, scale = 0.125)
        getitem_85: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_5[0]
        getitem_86: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_5[1]
        getitem_91: "u64[2]" = _scaled_dot_product_flash_attention_5[6]
        getitem_92: "u64[]" = _scaled_dot_product_flash_attention_5[7];  _scaled_dot_product_flash_attention_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_44: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_65: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_44, [4, 64, 768]);  permute_44 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_84: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        permute_45: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        view_66: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_65, [256, 768]);  view_65 = None
        mm_21: "bf16[256, 768]" = torch.ops.aten.mm.default(view_66, permute_45);  view_66 = None
        view_67: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_21, [4, 64, 768]);  mm_21 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_27: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_25, view_67);  add_25 = view_67 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 64, 1]" = var_mean_11[0]
        getitem_95: "f32[4, 64, 1]" = var_mean_11[1];  var_mean_11 = None
        add_28: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_11: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_95);  getitem_95 = None
        mul_37: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_38: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_37)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_87: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_88: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        permute_46: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        view_68: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_88, [256, 768]);  convert_element_type_88 = None
        mm_22: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_68, permute_46)
        view_69: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_22, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_91: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_39: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_40: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_5: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_29: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_29);  mul_39 = add_29 = None
        convert_element_type_92: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_93: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_47: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        view_70: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_92, [256, 3072]);  convert_element_type_92 = None
        mm_23: "bf16[256, 768]" = torch.ops.aten.mm.default(view_70, permute_47)
        view_71: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_23, [4, 64, 768]);  mm_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_30: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_27, view_71);  add_27 = view_71 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_96: "f32[4, 64, 1]" = var_mean_12[0]
        getitem_97: "f32[4, 64, 1]" = var_mean_12[1];  var_mean_12 = None
        add_31: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_12: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_12: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_97);  getitem_97 = None
        mul_42: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_43: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_40)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_96: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_97: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        permute_48: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_72: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_97, [256, 768]);  convert_element_type_97 = None
        mm_24: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_72, permute_48)
        view_73: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_24, [4, 64, 2304]);  mm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_73, 768, 2);  view_73 = None
        getitem_98: "bf16[4, 64, 768]" = split_6[0]
        getitem_99: "bf16[4, 64, 768]" = split_6[1]
        getitem_100: "bf16[4, 64, 768]" = split_6[2];  split_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_74: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_99, [4, 64, 12, 64]);  getitem_99 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_49: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_75: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_98, [4, 64, 12, 64]);  getitem_98 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_50: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_76: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_100, [4, 64, 12, 64]);  getitem_100 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_51: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_50, permute_49, permute_51, 0.0, True, scale = 0.125)
        getitem_101: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_6[0]
        getitem_102: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_6[1]
        getitem_107: "u64[2]" = _scaled_dot_product_flash_attention_6[6]
        getitem_108: "u64[]" = _scaled_dot_product_flash_attention_6[7];  _scaled_dot_product_flash_attention_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_52: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_77: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_52, [4, 64, 768]);  permute_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_100: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_53: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        view_78: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_77, [256, 768]);  view_77 = None
        mm_25: "bf16[256, 768]" = torch.ops.aten.mm.default(view_78, permute_53);  view_78 = None
        view_79: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_25, [4, 64, 768]);  mm_25 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_32: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_30, view_79);  add_30 = view_79 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_110: "f32[4, 64, 1]" = var_mean_13[0]
        getitem_111: "f32[4, 64, 1]" = var_mean_13[1];  var_mean_13 = None
        add_33: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_13: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_111);  getitem_111 = None
        mul_44: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_45: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_43)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_103: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_104: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        permute_54: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        view_80: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_104, [256, 768]);  convert_element_type_104 = None
        mm_26: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_80, permute_54)
        view_81: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_26, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_107: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_46: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.5)
        mul_47: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.7071067811865476);  convert_element_type_107 = None
        erf_6: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_34: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_34);  mul_46 = add_34 = None
        convert_element_type_108: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_109: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_55: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        view_82: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_108, [256, 3072]);  convert_element_type_108 = None
        mm_27: "bf16[256, 768]" = torch.ops.aten.mm.default(view_82, permute_55)
        view_83: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_27, [4, 64, 768]);  mm_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_35: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_32, view_83);  add_32 = view_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_112: "f32[4, 64, 1]" = var_mean_14[0]
        getitem_113: "f32[4, 64, 1]" = var_mean_14[1];  var_mean_14 = None
        add_36: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_14: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_113);  getitem_113 = None
        mul_49: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_50: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_46)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_112: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_113: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None
        permute_56: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        view_84: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_113, [256, 768]);  convert_element_type_113 = None
        mm_28: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_84, permute_56)
        view_85: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_28, [4, 64, 2304]);  mm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_85, 768, 2);  view_85 = None
        getitem_114: "bf16[4, 64, 768]" = split_7[0]
        getitem_115: "bf16[4, 64, 768]" = split_7[1]
        getitem_116: "bf16[4, 64, 768]" = split_7[2];  split_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_86: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_115, [4, 64, 12, 64]);  getitem_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_57: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_87: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [4, 64, 12, 64]);  getitem_114 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_58: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_88: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_116, [4, 64, 12, 64]);  getitem_116 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_59: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_58, permute_57, permute_59, 0.0, True, scale = 0.125)
        getitem_117: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_7[0]
        getitem_118: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_7[1]
        getitem_123: "u64[2]" = _scaled_dot_product_flash_attention_7[6]
        getitem_124: "u64[]" = _scaled_dot_product_flash_attention_7[7];  _scaled_dot_product_flash_attention_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_60: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_89: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_60, [4, 64, 768]);  permute_60 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_116: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_61: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        view_90: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_89, [256, 768]);  view_89 = None
        mm_29: "bf16[256, 768]" = torch.ops.aten.mm.default(view_90, permute_61);  view_90 = None
        view_91: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_29, [4, 64, 768]);  mm_29 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_37: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_35, view_91);  add_35 = view_91 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_126: "f32[4, 64, 1]" = var_mean_15[0]
        getitem_127: "f32[4, 64, 1]" = var_mean_15[1];  var_mean_15 = None
        add_38: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_15: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_127);  getitem_127 = None
        mul_51: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_52: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_49)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_119: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_120: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        permute_62: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        view_92: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_120, [256, 768]);  convert_element_type_120 = None
        mm_30: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_92, permute_62)
        view_93: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_30, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_123: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_53: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.5)
        mul_54: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476);  convert_element_type_123 = None
        erf_7: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_39: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_39);  mul_53 = add_39 = None
        convert_element_type_124: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_125: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_63: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        view_94: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_124, [256, 3072]);  convert_element_type_124 = None
        mm_31: "bf16[256, 768]" = torch.ops.aten.mm.default(view_94, permute_63)
        view_95: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_31, [4, 64, 768]);  mm_31 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_40: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_37, view_95);  add_37 = view_95 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_128: "f32[4, 64, 1]" = var_mean_16[0]
        getitem_129: "f32[4, 64, 1]" = var_mean_16[1];  var_mean_16 = None
        add_41: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_16: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_16: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_129);  getitem_129 = None
        mul_56: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_57: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_52)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_128: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_129: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        permute_64: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        view_96: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_129, [256, 768]);  convert_element_type_129 = None
        mm_32: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_96, permute_64)
        view_97: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_32, [4, 64, 2304]);  mm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_97, 768, 2);  view_97 = None
        getitem_130: "bf16[4, 64, 768]" = split_8[0]
        getitem_131: "bf16[4, 64, 768]" = split_8[1]
        getitem_132: "bf16[4, 64, 768]" = split_8[2];  split_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_98: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_131, [4, 64, 12, 64]);  getitem_131 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_65: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_99: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_130, [4, 64, 12, 64]);  getitem_130 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_66: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_100: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_132, [4, 64, 12, 64]);  getitem_132 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_67: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_66, permute_65, permute_67, 0.0, True, scale = 0.125)
        getitem_133: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_8[0]
        getitem_134: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_8[1]
        getitem_139: "u64[2]" = _scaled_dot_product_flash_attention_8[6]
        getitem_140: "u64[]" = _scaled_dot_product_flash_attention_8[7];  _scaled_dot_product_flash_attention_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_68: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_101: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_68, [4, 64, 768]);  permute_68 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_132: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        permute_69: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        view_102: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_101, [256, 768]);  view_101 = None
        mm_33: "bf16[256, 768]" = torch.ops.aten.mm.default(view_102, permute_69);  view_102 = None
        view_103: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_33, [4, 64, 768]);  mm_33 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_42: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_40, view_103);  add_40 = view_103 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_142: "f32[4, 64, 1]" = var_mean_17[0]
        getitem_143: "f32[4, 64, 1]" = var_mean_17[1];  var_mean_17 = None
        add_43: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_17: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_17: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_143);  getitem_143 = None
        mul_58: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_59: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_55)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_135: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_136: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        permute_70: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        view_104: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_136, [256, 768]);  convert_element_type_136 = None
        mm_34: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_104, permute_70)
        view_105: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_34, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_139: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_60: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_61: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_8: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_44: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_44);  mul_60 = add_44 = None
        convert_element_type_140: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_141: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        permute_71: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        view_106: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_140, [256, 3072]);  convert_element_type_140 = None
        mm_35: "bf16[256, 768]" = torch.ops.aten.mm.default(view_106, permute_71)
        view_107: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_35, [4, 64, 768]);  mm_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_45: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_42, view_107);  add_42 = view_107 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_144: "f32[4, 64, 1]" = var_mean_18[0]
        getitem_145: "f32[4, 64, 1]" = var_mean_18[1];  var_mean_18 = None
        add_46: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_18: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_145);  getitem_145 = None
        mul_63: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_64: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_58)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_144: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_145: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        permute_72: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        view_108: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_145, [256, 768]);  convert_element_type_145 = None
        mm_36: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_108, permute_72)
        view_109: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_36, [4, 64, 2304]);  mm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_109, 768, 2);  view_109 = None
        getitem_146: "bf16[4, 64, 768]" = split_9[0]
        getitem_147: "bf16[4, 64, 768]" = split_9[1]
        getitem_148: "bf16[4, 64, 768]" = split_9[2];  split_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_110: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_147, [4, 64, 12, 64]);  getitem_147 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_73: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_111: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_146, [4, 64, 12, 64]);  getitem_146 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_74: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_112: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_148, [4, 64, 12, 64]);  getitem_148 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_75: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_74, permute_73, permute_75, 0.0, True, scale = 0.125)
        getitem_149: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_9[0]
        getitem_150: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_9[1]
        getitem_155: "u64[2]" = _scaled_dot_product_flash_attention_9[6]
        getitem_156: "u64[]" = _scaled_dot_product_flash_attention_9[7];  _scaled_dot_product_flash_attention_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_76: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_113: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_76, [4, 64, 768]);  permute_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_148: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        permute_77: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_148, [1, 0]);  convert_element_type_148 = None
        view_114: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_113, [256, 768]);  view_113 = None
        mm_37: "bf16[256, 768]" = torch.ops.aten.mm.default(view_114, permute_77);  view_114 = None
        view_115: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_37, [4, 64, 768]);  mm_37 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_47: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_45, view_115);  add_45 = view_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_158: "f32[4, 64, 1]" = var_mean_19[0]
        getitem_159: "f32[4, 64, 1]" = var_mean_19[1];  var_mean_19 = None
        add_48: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_19: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_19: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_159);  getitem_159 = None
        mul_65: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_66: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_61)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_151: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_152: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        permute_78: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        view_116: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_152, [256, 768]);  convert_element_type_152 = None
        mm_38: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_116, permute_78)
        view_117: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_38, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_155: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_67: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.5)
        mul_68: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476);  convert_element_type_155 = None
        erf_9: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_49: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_49);  mul_67 = add_49 = None
        convert_element_type_156: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_157: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        permute_79: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_157, [1, 0]);  convert_element_type_157 = None
        view_118: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_156, [256, 3072]);  convert_element_type_156 = None
        mm_39: "bf16[256, 768]" = torch.ops.aten.mm.default(view_118, permute_79)
        view_119: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_39, [4, 64, 768]);  mm_39 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_50: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_47, view_119);  add_47 = view_119 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_160: "f32[4, 64, 1]" = var_mean_20[0]
        getitem_161: "f32[4, 64, 1]" = var_mean_20[1];  var_mean_20 = None
        add_51: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_20: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_20: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_161);  getitem_161 = None
        mul_70: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_71: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_64)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_160: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_161: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        permute_80: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_120: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_161, [256, 768]);  convert_element_type_161 = None
        mm_40: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_120, permute_80)
        view_121: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_40, [4, 64, 2304]);  mm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_121, 768, 2);  view_121 = None
        getitem_162: "bf16[4, 64, 768]" = split_10[0]
        getitem_163: "bf16[4, 64, 768]" = split_10[1]
        getitem_164: "bf16[4, 64, 768]" = split_10[2];  split_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_122: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_163, [4, 64, 12, 64]);  getitem_163 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_81: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_123: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_162, [4, 64, 12, 64]);  getitem_162 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_82: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_124: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_164, [4, 64, 12, 64]);  getitem_164 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_83: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_82, permute_81, permute_83, 0.0, True, scale = 0.125)
        getitem_165: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_10[0]
        getitem_166: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_10[1]
        getitem_171: "u64[2]" = _scaled_dot_product_flash_attention_10[6]
        getitem_172: "u64[]" = _scaled_dot_product_flash_attention_10[7];  _scaled_dot_product_flash_attention_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_84: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_125: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_84, [4, 64, 768]);  permute_84 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_164: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        permute_85: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        view_126: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_125, [256, 768]);  view_125 = None
        mm_41: "bf16[256, 768]" = torch.ops.aten.mm.default(view_126, permute_85);  view_126 = None
        view_127: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_41, [4, 64, 768]);  mm_41 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_52: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_50, view_127);  add_50 = view_127 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_174: "f32[4, 64, 1]" = var_mean_21[0]
        getitem_175: "f32[4, 64, 1]" = var_mean_21[1];  var_mean_21 = None
        add_53: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_21: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_21: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_175);  getitem_175 = None
        mul_72: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_73: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_67)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_167: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_168: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        view_128: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_168, [256, 768]);  convert_element_type_168 = None
        mm_42: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_128, permute_86)
        view_129: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_42, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_171: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_74: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.5)
        mul_75: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.7071067811865476);  convert_element_type_171 = None
        erf_10: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_54);  mul_74 = add_54 = None
        convert_element_type_172: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_173: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        permute_87: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        view_130: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_172, [256, 3072]);  convert_element_type_172 = None
        mm_43: "bf16[256, 768]" = torch.ops.aten.mm.default(view_130, permute_87)
        view_131: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_43, [4, 64, 768]);  mm_43 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_55: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_52, view_131);  add_52 = view_131 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_176: "f32[4, 64, 1]" = var_mean_22[0]
        getitem_177: "f32[4, 64, 1]" = var_mean_22[1];  var_mean_22 = None
        add_56: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_22: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_22: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_177);  getitem_177 = None
        mul_77: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_78: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_70)
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_176: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_177: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        permute_88: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        view_132: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_177, [256, 768]);  convert_element_type_177 = None
        mm_44: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_132, permute_88)
        view_133: "bf16[4, 64, 2304]" = torch.ops.aten.reshape.default(mm_44, [4, 64, 2304]);  mm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_133, 768, 2);  view_133 = None
        getitem_178: "bf16[4, 64, 768]" = split_11[0]
        getitem_179: "bf16[4, 64, 768]" = split_11[1]
        getitem_180: "bf16[4, 64, 768]" = split_11[2];  split_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_134: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_179, [4, 64, 12, 64]);  getitem_179 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_89: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_135: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_178, [4, 64, 12, 64]);  getitem_178 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_90: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_136: "bf16[4, 64, 12, 64]" = torch.ops.aten.reshape.default(getitem_180, [4, 64, 12, 64]);  getitem_180 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_91: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_90, permute_89, permute_91, 0.0, True, scale = 0.125)
        getitem_181: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_11[0]
        getitem_182: "f32[4, 12, 64]" = _scaled_dot_product_flash_attention_11[1]
        getitem_187: "u64[2]" = _scaled_dot_product_flash_attention_11[6]
        getitem_188: "u64[]" = _scaled_dot_product_flash_attention_11[7];  _scaled_dot_product_flash_attention_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_92: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_137: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(permute_92, [4, 64, 768]);  permute_92 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_180: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        permute_93: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        view_138: "bf16[256, 768]" = torch.ops.aten.reshape.default(view_137, [256, 768]);  view_137 = None
        mm_45: "bf16[256, 768]" = torch.ops.aten.mm.default(view_138, permute_93);  view_138 = None
        view_139: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_45, [4, 64, 768]);  mm_45 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_57: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_55, view_139);  add_55 = view_139 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_190: "f32[4, 64, 1]" = var_mean_23[0]
        getitem_191: "f32[4, 64, 1]" = var_mean_23[1];  var_mean_23 = None
        add_58: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_23: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_23: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_191);  getitem_191 = None
        mul_79: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_80: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_73)
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_183: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_184: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        permute_94: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_183, [1, 0]);  convert_element_type_183 = None
        view_140: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_184, [256, 768]);  convert_element_type_184 = None
        mm_46: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_140, permute_94)
        view_141: "bf16[4, 64, 3072]" = torch.ops.aten.reshape.default(mm_46, [4, 64, 3072])
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_187: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_81: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5)
        mul_82: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476);  convert_element_type_187 = None
        erf_11: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_59: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_59);  mul_81 = add_59 = None
        convert_element_type_188: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_189: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_95: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        view_142: "bf16[256, 3072]" = torch.ops.aten.reshape.default(convert_element_type_188, [256, 3072]);  convert_element_type_188 = None
        mm_47: "bf16[256, 768]" = torch.ops.aten.mm.default(view_142, permute_95)
        view_143: "bf16[4, 64, 768]" = torch.ops.aten.reshape.default(mm_47, [4, 64, 768]);  mm_47 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_60: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_57, view_143);  add_57 = view_143 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_192: "f32[4, 64, 1]" = var_mean_24[0]
        getitem_193: "f32[4, 64, 1]" = var_mean_24[1];  var_mean_24 = None
        add_61: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_24: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_24: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_193);  add_60 = getitem_193 = None
        mul_84: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_85: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_76)
        
         # File: /home/ubuntu/nanoGPT/model.py:327 in forward, code: logits = self.lm_head(x)
        convert_element_type_192: "bf16[65, 768]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_193: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        permute_96: "bf16[768, 65]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        view_144: "bf16[256, 768]" = torch.ops.aten.reshape.default(convert_element_type_193, [256, 768]);  convert_element_type_193 = None
        mm_48: "bf16[256, 65]" = torch.ops.aten.mm.default(view_144, permute_96)
        view_145: "bf16[4, 64, 65]" = torch.ops.aten.reshape.default(mm_48, [4, 64, 65]);  mm_48 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        view_146: "bf16[256, 65]" = torch.ops.aten.reshape.default(view_145, [-1, 65])
        view_147: "i64[256]" = torch.ops.aten.reshape.default(primals_77, [-1])
        
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        convert_element_type_196: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(convert_element_type_196, 1)
        getitem_194: "f32[256, 1]" = prepare_softmax_online_default[0]
        getitem_195: "f32[256, 1]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[256, 65]" = torch.ops.aten.sub.Tensor(convert_element_type_196, getitem_194);  convert_element_type_196 = None
        exp_default: "f32[256, 65]" = torch.ops.aten.exp.default(sub_tensor);  exp_default = None
        
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        log: "f32[256, 1]" = torch.ops.aten.log.default(getitem_195);  getitem_195 = None
        sub_26: "f32[256, 65]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = None
        convert_element_type_197: "bf16[256, 65]" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        convert_element_type_198: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(convert_element_type_197, torch.float32);  convert_element_type_197 = None
        ne: "b8[256]" = torch.ops.aten.ne.Scalar(view_147, -1)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[256]" = torch.ops.aten.where.self(ne, view_147, full_default);  view_147 = full_default = None
        unsqueeze: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[256, 1]" = torch.ops.aten.gather.default(convert_element_type_198, 1, unsqueeze);  convert_element_type_198 = unsqueeze = None
        squeeze: "f32[256]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[256]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[256]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_199: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_199);  sum_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:327 in forward, code: logits = self.lm_head(x)
        permute_99: "bf16[65, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_2: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_103: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_107: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_3: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_111: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_119: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_4: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_123: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_127: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_5: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_131: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_139: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_6: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_143: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_147: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_7: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_151: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_159: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_8: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_163: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_167: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_9: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_171: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_179: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_10: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_183: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_187: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_11: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_191: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_199: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_12: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_203: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_207: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_13: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_211: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_219: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_14: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_223: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_227: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_15: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_231: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_239: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_16: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_243: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_247: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_17: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_251: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_259: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_18: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_263: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_267: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_19: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_271: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_279: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_20: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_283: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_287: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_21: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_291: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_299: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_22: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_303: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_307: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_23: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_311: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_319: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_24: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        permute_323: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        permute_327: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        div_25: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        permute_331: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        permute_339: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (view_145, div, primals_1, primals_4, primals_7, primals_10, primals_13, primals_16, primals_19, primals_22, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_77, iota, embedding, embedding_1, getitem_1, rsqrt, view, permute_1, permute_2, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mul_2, view_8, mm_2, view_10, mul_7, view_12, permute_9, permute_10, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_9, view_20, mm_6, view_22, mul_14, view_24, permute_17, permute_18, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_16, view_32, mm_10, view_34, mul_21, view_36, permute_25, permute_26, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_23, view_44, mm_14, view_46, mul_28, view_48, permute_33, permute_34, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_30, view_56, mm_18, view_58, mul_35, view_60, permute_41, permute_42, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_37, view_68, mm_22, view_70, mul_42, view_72, permute_49, permute_50, permute_51, getitem_101, getitem_102, getitem_107, getitem_108, mul_44, view_80, mm_26, view_82, mul_49, view_84, permute_57, permute_58, permute_59, getitem_117, getitem_118, getitem_123, getitem_124, mul_51, view_92, mm_30, view_94, mul_56, view_96, permute_65, permute_66, permute_67, getitem_133, getitem_134, getitem_139, getitem_140, mul_58, view_104, mm_34, view_106, mul_63, view_108, permute_73, permute_74, permute_75, getitem_149, getitem_150, getitem_155, getitem_156, mul_65, view_116, mm_38, view_118, mul_70, view_120, permute_81, permute_82, permute_83, getitem_165, getitem_166, getitem_171, getitem_172, mul_72, view_128, mm_42, view_130, mul_77, view_132, permute_89, permute_90, permute_91, getitem_181, getitem_182, getitem_187, getitem_188, mul_79, view_140, mm_46, view_142, mul_84, view_144, view_145, getitem_194, log, convert_element_type_199, permute_99, div_2, permute_103, permute_107, div_3, permute_111, permute_119, div_4, permute_123, permute_127, div_5, permute_131, permute_139, div_6, permute_143, permute_147, div_7, permute_151, permute_159, div_8, permute_163, permute_167, div_9, permute_171, permute_179, div_10, permute_183, permute_187, div_11, permute_191, permute_199, div_12, permute_203, permute_207, div_13, permute_211, permute_219, div_14, permute_223, permute_227, div_15, permute_231, permute_239, div_16, permute_243, permute_247, div_17, permute_251, permute_259, div_18, permute_263, permute_267, div_19, permute_271, permute_279, div_20, permute_283, permute_287, div_21, permute_291, permute_299, div_22, permute_303, permute_307, div_23, permute_311, permute_319, div_24, permute_323, permute_327, div_25, permute_331, permute_339)
        