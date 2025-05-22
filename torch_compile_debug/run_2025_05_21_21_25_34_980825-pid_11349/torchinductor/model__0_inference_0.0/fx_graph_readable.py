class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 64]", arg1_1: "f32[65, 768]", arg2_1: "f32[64, 768]", arg3_1: "f32[768]", arg4_1: "f32[2304, 768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[3072, 768]", arg8_1: "f32[768, 3072]", arg9_1: "f32[768]", arg10_1: "f32[2304, 768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[3072, 768]", arg14_1: "f32[768, 3072]", arg15_1: "f32[768]", arg16_1: "f32[2304, 768]", arg17_1: "f32[768, 768]", arg18_1: "f32[768]", arg19_1: "f32[3072, 768]", arg20_1: "f32[768, 3072]", arg21_1: "f32[768]", arg22_1: "f32[2304, 768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[3072, 768]", arg26_1: "f32[768, 3072]", arg27_1: "f32[768]", arg28_1: "f32[2304, 768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[768, 3072]", arg33_1: "f32[768]", arg34_1: "f32[2304, 768]", arg35_1: "f32[768, 768]", arg36_1: "f32[768]", arg37_1: "f32[3072, 768]", arg38_1: "f32[768, 3072]", arg39_1: "f32[768]", arg40_1: "f32[2304, 768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[3072, 768]", arg44_1: "f32[768, 3072]", arg45_1: "f32[768]", arg46_1: "f32[2304, 768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[768, 3072]", arg51_1: "f32[768]", arg52_1: "f32[2304, 768]", arg53_1: "f32[768, 768]", arg54_1: "f32[768]", arg55_1: "f32[3072, 768]", arg56_1: "f32[768, 3072]", arg57_1: "f32[768]", arg58_1: "f32[2304, 768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[3072, 768]", arg62_1: "f32[768, 3072]", arg63_1: "f32[768]", arg64_1: "f32[2304, 768]", arg65_1: "f32[768, 768]", arg66_1: "f32[768]", arg67_1: "f32[3072, 768]", arg68_1: "f32[768, 3072]", arg69_1: "f32[768]", arg70_1: "f32[2304, 768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[3072, 768]", arg74_1: "f32[768, 3072]", arg75_1: "f32[768]", arg76_1: "i64[4, 64]"):
         # File: /home/ubuntu/nanoGPT/model.py:306 in forward, code: pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        iota: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/ubuntu/nanoGPT/model.py:312 in forward, code: tok_emb = self.transformer.wte(idx)
        embedding: "f32[4, 64, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:315 in forward, code: pos_emb = self.transformer.wpe(pos)
        embedding_1: "f32[64, 768]" = torch.ops.aten.embedding.default(arg2_1, iota);  arg2_1 = iota = None
        
         # File: /home/ubuntu/nanoGPT/model.py:316 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        add: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 64, 1]" = var_mean[0]
        getitem_1: "f32[4, 64, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_1: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_1, [256, 768]);  convert_element_type_1 = None
        mm: "bf16[256, 2304]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm, [4, 64, 2304]);  mm = None
        split = torch.ops.aten.split.Tensor(view_1, 768, 2);  view_1 = None
        getitem_2: "bf16[4, 64, 768]" = split[0]
        getitem_3: "bf16[4, 64, 768]" = split[1]
        getitem_4: "bf16[4, 64, 768]" = split[2];  split = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_2: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_3, [4, 64, 12, 64]);  getitem_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_1: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_3: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_2, [4, 64, 12, 64]);  getitem_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_2: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_4: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_4, [4, 64, 12, 64]);  getitem_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_3: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_2, permute_1, permute_3, 0.0, True, scale = 0.125);  permute_2 = permute_1 = permute_3 = None
        getitem_5: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_4: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_5: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_4, [4, 64, 768]);  permute_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_4: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        permute_5: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        view_6: "bf16[256, 768]" = torch.ops.aten.view.default(view_5, [256, 768]);  view_5 = None
        mm_1: "bf16[256, 768]" = torch.ops.aten.mm.default(view_6, permute_5);  view_6 = permute_5 = None
        view_7: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_1, [4, 64, 768]);  mm_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_2: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add, view_7);  add = view_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 64, 1]" = var_mean_1[0]
        getitem_15: "f32[4, 64, 1]" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_1: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_15);  getitem_15 = None
        mul_2: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_7: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_8: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_6: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        view_8: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_8, [256, 768]);  convert_element_type_8 = None
        mm_2: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_8, permute_6);  view_8 = permute_6 = None
        view_9: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_2, [4, 64, 3072]);  mm_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_11: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_4: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5)
        mul_5: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476);  convert_element_type_11 = None
        erf: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_12: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_13: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_7: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        view_10: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_12, [256, 3072]);  convert_element_type_12 = None
        mm_3: "bf16[256, 768]" = torch.ops.aten.mm.default(view_10, permute_7);  view_10 = permute_7 = None
        view_11: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_3, [4, 64, 768]);  mm_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_5: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 64, 1]" = var_mean_2[0]
        getitem_17: "f32[4, 64, 1]" = var_mean_2[1];  var_mean_2 = None
        add_6: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_2: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_17);  getitem_17 = None
        mul_7: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_8: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg9_1);  mul_7 = arg9_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_16: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_17: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        permute_8: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        view_12: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_17, [256, 768]);  convert_element_type_17 = None
        mm_4: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_12, permute_8);  view_12 = permute_8 = None
        view_13: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_4, [4, 64, 2304]);  mm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_13, 768, 2);  view_13 = None
        getitem_18: "bf16[4, 64, 768]" = split_1[0]
        getitem_19: "bf16[4, 64, 768]" = split_1[1]
        getitem_20: "bf16[4, 64, 768]" = split_1[2];  split_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_14: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_19, [4, 64, 12, 64]);  getitem_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_9: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_15: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_18, [4, 64, 12, 64]);  getitem_18 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_10: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_16: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_20, [4, 64, 12, 64]);  getitem_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_11: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_10, permute_9, permute_11, 0.0, True, scale = 0.125);  permute_10 = permute_9 = permute_11 = None
        getitem_21: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_12: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_17: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_12, [4, 64, 768]);  permute_12 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_20: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        view_18: "bf16[256, 768]" = torch.ops.aten.view.default(view_17, [256, 768]);  view_17 = None
        mm_5: "bf16[256, 768]" = torch.ops.aten.mm.default(view_18, permute_13);  view_18 = permute_13 = None
        view_19: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_5, [4, 64, 768]);  mm_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_7: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_5, view_19);  add_5 = view_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 64, 1]" = var_mean_3[0]
        getitem_31: "f32[4, 64, 1]" = var_mean_3[1];  var_mean_3 = None
        add_8: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_3: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_31);  getitem_31 = None
        mul_9: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_10: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg12_1);  mul_9 = arg12_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_23: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_24: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        permute_14: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        view_20: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_24, [256, 768]);  convert_element_type_24 = None
        mm_6: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_20, permute_14);  view_20 = permute_14 = None
        view_21: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_6, [4, 64, 3072]);  mm_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_27: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_11: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.5)
        mul_12: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 0.7071067811865476);  convert_element_type_27 = None
        erf_1: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = add_9 = None
        convert_element_type_28: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_29: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_15: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_22: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_28, [256, 3072]);  convert_element_type_28 = None
        mm_7: "bf16[256, 768]" = torch.ops.aten.mm.default(view_22, permute_15);  view_22 = permute_15 = None
        view_23: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_7, [4, 64, 768]);  mm_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_10: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_7, view_23);  add_7 = view_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 64, 1]" = var_mean_4[0]
        getitem_33: "f32[4, 64, 1]" = var_mean_4[1];  var_mean_4 = None
        add_11: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_4: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_33);  getitem_33 = None
        mul_14: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_15: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg15_1);  mul_14 = arg15_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_32: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_33: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        permute_16: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        view_24: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_33, [256, 768]);  convert_element_type_33 = None
        mm_8: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_24, permute_16);  view_24 = permute_16 = None
        view_25: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_8, [4, 64, 2304]);  mm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_25, 768, 2);  view_25 = None
        getitem_34: "bf16[4, 64, 768]" = split_2[0]
        getitem_35: "bf16[4, 64, 768]" = split_2[1]
        getitem_36: "bf16[4, 64, 768]" = split_2[2];  split_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_26: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_35, [4, 64, 12, 64]);  getitem_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_17: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_27: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_34, [4, 64, 12, 64]);  getitem_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_18: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_28: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_36, [4, 64, 12, 64]);  getitem_36 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_19: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_18, permute_17, permute_19, 0.0, True, scale = 0.125);  permute_18 = permute_17 = permute_19 = None
        getitem_37: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_20: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
        view_29: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_20, [4, 64, 768]);  permute_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_36: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        permute_21: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        view_30: "bf16[256, 768]" = torch.ops.aten.view.default(view_29, [256, 768]);  view_29 = None
        mm_9: "bf16[256, 768]" = torch.ops.aten.mm.default(view_30, permute_21);  view_30 = permute_21 = None
        view_31: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_9, [4, 64, 768]);  mm_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_12: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_10, view_31);  add_10 = view_31 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 64, 1]" = var_mean_5[0]
        getitem_47: "f32[4, 64, 1]" = var_mean_5[1];  var_mean_5 = None
        add_13: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_5: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_47);  getitem_47 = None
        mul_16: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_17: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg18_1);  mul_16 = arg18_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_39: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_40: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        permute_22: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        view_32: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_40, [256, 768]);  convert_element_type_40 = None
        mm_10: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_32, permute_22);  view_32 = permute_22 = None
        view_33: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_10, [4, 64, 3072]);  mm_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_43: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_18: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_19: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_2: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
        convert_element_type_44: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_45: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        permute_23: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        view_34: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_44, [256, 3072]);  convert_element_type_44 = None
        mm_11: "bf16[256, 768]" = torch.ops.aten.mm.default(view_34, permute_23);  view_34 = permute_23 = None
        view_35: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_11, [4, 64, 768]);  mm_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_15: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_12, view_35);  add_12 = view_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 64, 1]" = var_mean_6[0]
        getitem_49: "f32[4, 64, 1]" = var_mean_6[1];  var_mean_6 = None
        add_16: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_6: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_49);  getitem_49 = None
        mul_21: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_22: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg21_1);  mul_21 = arg21_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_48: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_49: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        permute_24: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        view_36: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_49, [256, 768]);  convert_element_type_49 = None
        mm_12: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_36, permute_24);  view_36 = permute_24 = None
        view_37: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_12, [4, 64, 2304]);  mm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_37, 768, 2);  view_37 = None
        getitem_50: "bf16[4, 64, 768]" = split_3[0]
        getitem_51: "bf16[4, 64, 768]" = split_3[1]
        getitem_52: "bf16[4, 64, 768]" = split_3[2];  split_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_38: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_51, [4, 64, 12, 64]);  getitem_51 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_25: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_39: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_50, [4, 64, 12, 64]);  getitem_50 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_26: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_40: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_52, [4, 64, 12, 64]);  getitem_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_27: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_26, permute_25, permute_27, 0.0, True, scale = 0.125);  permute_26 = permute_25 = permute_27 = None
        getitem_53: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_28: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_41: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_28, [4, 64, 768]);  permute_28 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_52: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_29: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        view_42: "bf16[256, 768]" = torch.ops.aten.view.default(view_41, [256, 768]);  view_41 = None
        mm_13: "bf16[256, 768]" = torch.ops.aten.mm.default(view_42, permute_29);  view_42 = permute_29 = None
        view_43: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_13, [4, 64, 768]);  mm_13 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_17: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_15, view_43);  add_15 = view_43 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 64, 1]" = var_mean_7[0]
        getitem_63: "f32[4, 64, 1]" = var_mean_7[1];  var_mean_7 = None
        add_18: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_7: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_63);  getitem_63 = None
        mul_23: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_24: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg24_1);  mul_23 = arg24_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_55: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_56: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        permute_30: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        view_44: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_56, [256, 768]);  convert_element_type_56 = None
        mm_14: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_44, permute_30);  view_44 = permute_30 = None
        view_45: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_14, [4, 64, 3072]);  mm_14 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_59: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_25: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.5)
        mul_26: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476);  convert_element_type_59 = None
        erf_3: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_19: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_19);  mul_25 = add_19 = None
        convert_element_type_60: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_61: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        permute_31: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        view_46: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_60, [256, 3072]);  convert_element_type_60 = None
        mm_15: "bf16[256, 768]" = torch.ops.aten.mm.default(view_46, permute_31);  view_46 = permute_31 = None
        view_47: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_15, [4, 64, 768]);  mm_15 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_20: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_17, view_47);  add_17 = view_47 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 64, 1]" = var_mean_8[0]
        getitem_65: "f32[4, 64, 1]" = var_mean_8[1];  var_mean_8 = None
        add_21: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_8: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_8: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_65);  getitem_65 = None
        mul_28: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_29: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg27_1);  mul_28 = arg27_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_64: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_65: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        permute_32: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_48: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_65, [256, 768]);  convert_element_type_65 = None
        mm_16: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_48, permute_32);  view_48 = permute_32 = None
        view_49: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_16, [4, 64, 2304]);  mm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_49, 768, 2);  view_49 = None
        getitem_66: "bf16[4, 64, 768]" = split_4[0]
        getitem_67: "bf16[4, 64, 768]" = split_4[1]
        getitem_68: "bf16[4, 64, 768]" = split_4[2];  split_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_50: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_67, [4, 64, 12, 64]);  getitem_67 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_33: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_51: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_66, [4, 64, 12, 64]);  getitem_66 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_34: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_52: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_68, [4, 64, 12, 64]);  getitem_68 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_35: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_34, permute_33, permute_35, 0.0, True, scale = 0.125);  permute_34 = permute_33 = permute_35 = None
        getitem_69: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_36: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_53: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_36, [4, 64, 768]);  permute_36 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_68: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        permute_37: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_54: "bf16[256, 768]" = torch.ops.aten.view.default(view_53, [256, 768]);  view_53 = None
        mm_17: "bf16[256, 768]" = torch.ops.aten.mm.default(view_54, permute_37);  view_54 = permute_37 = None
        view_55: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_17, [4, 64, 768]);  mm_17 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_22: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_20, view_55);  add_20 = view_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 64, 1]" = var_mean_9[0]
        getitem_79: "f32[4, 64, 1]" = var_mean_9[1];  var_mean_9 = None
        add_23: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_9: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_9: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_79);  getitem_79 = None
        mul_30: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_31: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg30_1);  mul_30 = arg30_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_71: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_72: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        permute_38: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        view_56: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_72, [256, 768]);  convert_element_type_72 = None
        mm_18: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_56, permute_38);  view_56 = permute_38 = None
        view_57: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_18, [4, 64, 3072]);  mm_18 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_75: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_32: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.5)
        mul_33: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476);  convert_element_type_75 = None
        erf_4: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_24: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_24);  mul_32 = add_24 = None
        convert_element_type_76: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_77: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        permute_39: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        view_58: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_76, [256, 3072]);  convert_element_type_76 = None
        mm_19: "bf16[256, 768]" = torch.ops.aten.mm.default(view_58, permute_39);  view_58 = permute_39 = None
        view_59: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_19, [4, 64, 768]);  mm_19 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_25: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_22, view_59);  add_22 = view_59 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 64, 1]" = var_mean_10[0]
        getitem_81: "f32[4, 64, 1]" = var_mean_10[1];  var_mean_10 = None
        add_26: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_10: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_81);  getitem_81 = None
        mul_35: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_36: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg33_1);  mul_35 = arg33_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_80: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_81: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        permute_40: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_60: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_81, [256, 768]);  convert_element_type_81 = None
        mm_20: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_60, permute_40);  view_60 = permute_40 = None
        view_61: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_20, [4, 64, 2304]);  mm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_61, 768, 2);  view_61 = None
        getitem_82: "bf16[4, 64, 768]" = split_5[0]
        getitem_83: "bf16[4, 64, 768]" = split_5[1]
        getitem_84: "bf16[4, 64, 768]" = split_5[2];  split_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_62: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_83, [4, 64, 12, 64]);  getitem_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_41: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_63: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_82, [4, 64, 12, 64]);  getitem_82 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_42: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_64: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_84, [4, 64, 12, 64]);  getitem_84 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_43: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_42, permute_41, permute_43, 0.0, True, scale = 0.125);  permute_42 = permute_41 = permute_43 = None
        getitem_85: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_5[0];  _scaled_dot_product_flash_attention_5 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_44: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_65: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_44, [4, 64, 768]);  permute_44 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_84: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        permute_45: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        view_66: "bf16[256, 768]" = torch.ops.aten.view.default(view_65, [256, 768]);  view_65 = None
        mm_21: "bf16[256, 768]" = torch.ops.aten.mm.default(view_66, permute_45);  view_66 = permute_45 = None
        view_67: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_21, [4, 64, 768]);  mm_21 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_27: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_25, view_67);  add_25 = view_67 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 64, 1]" = var_mean_11[0]
        getitem_95: "f32[4, 64, 1]" = var_mean_11[1];  var_mean_11 = None
        add_28: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_11: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_95);  getitem_95 = None
        mul_37: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_38: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg36_1);  mul_37 = arg36_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_87: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convert_element_type_88: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        permute_46: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        view_68: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_88, [256, 768]);  convert_element_type_88 = None
        mm_22: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_68, permute_46);  view_68 = permute_46 = None
        view_69: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_22, [4, 64, 3072]);  mm_22 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_91: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_39: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_40: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_5: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_29: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_29);  mul_39 = add_29 = None
        convert_element_type_92: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_93: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        permute_47: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        view_70: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_92, [256, 3072]);  convert_element_type_92 = None
        mm_23: "bf16[256, 768]" = torch.ops.aten.mm.default(view_70, permute_47);  view_70 = permute_47 = None
        view_71: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_23, [4, 64, 768]);  mm_23 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_30: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_27, view_71);  add_27 = view_71 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_96: "f32[4, 64, 1]" = var_mean_12[0]
        getitem_97: "f32[4, 64, 1]" = var_mean_12[1];  var_mean_12 = None
        add_31: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_12: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_12: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_97);  getitem_97 = None
        mul_42: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_43: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg39_1);  mul_42 = arg39_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_96: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        convert_element_type_97: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        permute_48: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_72: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_97, [256, 768]);  convert_element_type_97 = None
        mm_24: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_72, permute_48);  view_72 = permute_48 = None
        view_73: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_24, [4, 64, 2304]);  mm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_73, 768, 2);  view_73 = None
        getitem_98: "bf16[4, 64, 768]" = split_6[0]
        getitem_99: "bf16[4, 64, 768]" = split_6[1]
        getitem_100: "bf16[4, 64, 768]" = split_6[2];  split_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_74: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_99, [4, 64, 12, 64]);  getitem_99 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_49: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_75: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_98, [4, 64, 12, 64]);  getitem_98 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_50: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_76: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_100, [4, 64, 12, 64]);  getitem_100 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_51: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_50, permute_49, permute_51, 0.0, True, scale = 0.125);  permute_50 = permute_49 = permute_51 = None
        getitem_101: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_6[0];  _scaled_dot_product_flash_attention_6 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_52: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        view_77: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_52, [4, 64, 768]);  permute_52 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_100: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        permute_53: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        view_78: "bf16[256, 768]" = torch.ops.aten.view.default(view_77, [256, 768]);  view_77 = None
        mm_25: "bf16[256, 768]" = torch.ops.aten.mm.default(view_78, permute_53);  view_78 = permute_53 = None
        view_79: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_25, [4, 64, 768]);  mm_25 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_32: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_30, view_79);  add_30 = view_79 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_110: "f32[4, 64, 1]" = var_mean_13[0]
        getitem_111: "f32[4, 64, 1]" = var_mean_13[1];  var_mean_13 = None
        add_33: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_13: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_111);  getitem_111 = None
        mul_44: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_45: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg42_1);  mul_44 = arg42_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_103: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        convert_element_type_104: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        permute_54: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        view_80: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_104, [256, 768]);  convert_element_type_104 = None
        mm_26: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_80, permute_54);  view_80 = permute_54 = None
        view_81: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_26, [4, 64, 3072]);  mm_26 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_107: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_46: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.5)
        mul_47: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 0.7071067811865476);  convert_element_type_107 = None
        erf_6: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_34: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_34);  mul_46 = add_34 = None
        convert_element_type_108: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_109: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        permute_55: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        view_82: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_108, [256, 3072]);  convert_element_type_108 = None
        mm_27: "bf16[256, 768]" = torch.ops.aten.mm.default(view_82, permute_55);  view_82 = permute_55 = None
        view_83: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_27, [4, 64, 768]);  mm_27 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_35: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_32, view_83);  add_32 = view_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_112: "f32[4, 64, 1]" = var_mean_14[0]
        getitem_113: "f32[4, 64, 1]" = var_mean_14[1];  var_mean_14 = None
        add_36: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_14: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_113);  getitem_113 = None
        mul_49: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_50: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg45_1);  mul_49 = arg45_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_112: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_113: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None
        permute_56: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        view_84: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_113, [256, 768]);  convert_element_type_113 = None
        mm_28: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_84, permute_56);  view_84 = permute_56 = None
        view_85: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_28, [4, 64, 2304]);  mm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_85, 768, 2);  view_85 = None
        getitem_114: "bf16[4, 64, 768]" = split_7[0]
        getitem_115: "bf16[4, 64, 768]" = split_7[1]
        getitem_116: "bf16[4, 64, 768]" = split_7[2];  split_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_86: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_115, [4, 64, 12, 64]);  getitem_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_57: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_87: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_114, [4, 64, 12, 64]);  getitem_114 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_58: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_88: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_116, [4, 64, 12, 64]);  getitem_116 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_59: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_58, permute_57, permute_59, 0.0, True, scale = 0.125);  permute_58 = permute_57 = permute_59 = None
        getitem_117: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_7[0];  _scaled_dot_product_flash_attention_7 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_60: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        view_89: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_60, [4, 64, 768]);  permute_60 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_116: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        permute_61: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        view_90: "bf16[256, 768]" = torch.ops.aten.view.default(view_89, [256, 768]);  view_89 = None
        mm_29: "bf16[256, 768]" = torch.ops.aten.mm.default(view_90, permute_61);  view_90 = permute_61 = None
        view_91: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_29, [4, 64, 768]);  mm_29 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_37: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_35, view_91);  add_35 = view_91 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_126: "f32[4, 64, 1]" = var_mean_15[0]
        getitem_127: "f32[4, 64, 1]" = var_mean_15[1];  var_mean_15 = None
        add_38: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_15: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_127);  getitem_127 = None
        mul_51: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_52: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg48_1);  mul_51 = arg48_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_119: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_120: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        permute_62: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        view_92: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_120, [256, 768]);  convert_element_type_120 = None
        mm_30: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_92, permute_62);  view_92 = permute_62 = None
        view_93: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_30, [4, 64, 3072]);  mm_30 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_123: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_53: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.5)
        mul_54: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476);  convert_element_type_123 = None
        erf_7: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_39: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_39);  mul_53 = add_39 = None
        convert_element_type_124: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_125: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        permute_63: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        view_94: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_124, [256, 3072]);  convert_element_type_124 = None
        mm_31: "bf16[256, 768]" = torch.ops.aten.mm.default(view_94, permute_63);  view_94 = permute_63 = None
        view_95: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_31, [4, 64, 768]);  mm_31 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_40: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_37, view_95);  add_37 = view_95 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_128: "f32[4, 64, 1]" = var_mean_16[0]
        getitem_129: "f32[4, 64, 1]" = var_mean_16[1];  var_mean_16 = None
        add_41: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_16: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_16: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_129);  getitem_129 = None
        mul_56: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_57: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg51_1);  mul_56 = arg51_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_128: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_129: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        permute_64: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        view_96: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_129, [256, 768]);  convert_element_type_129 = None
        mm_32: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_96, permute_64);  view_96 = permute_64 = None
        view_97: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_32, [4, 64, 2304]);  mm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_97, 768, 2);  view_97 = None
        getitem_130: "bf16[4, 64, 768]" = split_8[0]
        getitem_131: "bf16[4, 64, 768]" = split_8[1]
        getitem_132: "bf16[4, 64, 768]" = split_8[2];  split_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_98: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_131, [4, 64, 12, 64]);  getitem_131 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_65: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_99: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_130, [4, 64, 12, 64]);  getitem_130 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_66: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_100: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_132, [4, 64, 12, 64]);  getitem_132 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_67: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_66, permute_65, permute_67, 0.0, True, scale = 0.125);  permute_66 = permute_65 = permute_67 = None
        getitem_133: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_8[0];  _scaled_dot_product_flash_attention_8 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_68: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        view_101: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_68, [4, 64, 768]);  permute_68 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_132: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        permute_69: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        view_102: "bf16[256, 768]" = torch.ops.aten.view.default(view_101, [256, 768]);  view_101 = None
        mm_33: "bf16[256, 768]" = torch.ops.aten.mm.default(view_102, permute_69);  view_102 = permute_69 = None
        view_103: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_33, [4, 64, 768]);  mm_33 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_42: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_40, view_103);  add_40 = view_103 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_142: "f32[4, 64, 1]" = var_mean_17[0]
        getitem_143: "f32[4, 64, 1]" = var_mean_17[1];  var_mean_17 = None
        add_43: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_17: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_17: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_143);  getitem_143 = None
        mul_58: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_59: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg54_1);  mul_58 = arg54_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_135: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_136: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        permute_70: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        view_104: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_136, [256, 768]);  convert_element_type_136 = None
        mm_34: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_104, permute_70);  view_104 = permute_70 = None
        view_105: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_34, [4, 64, 3072]);  mm_34 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_139: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_60: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_61: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_8: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_44: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_44);  mul_60 = add_44 = None
        convert_element_type_140: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_141: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        permute_71: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        view_106: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_140, [256, 3072]);  convert_element_type_140 = None
        mm_35: "bf16[256, 768]" = torch.ops.aten.mm.default(view_106, permute_71);  view_106 = permute_71 = None
        view_107: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_35, [4, 64, 768]);  mm_35 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_45: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_42, view_107);  add_42 = view_107 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_144: "f32[4, 64, 1]" = var_mean_18[0]
        getitem_145: "f32[4, 64, 1]" = var_mean_18[1];  var_mean_18 = None
        add_46: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_18: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_145);  getitem_145 = None
        mul_63: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_64: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg57_1);  mul_63 = arg57_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_144: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_145: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        permute_72: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        view_108: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_145, [256, 768]);  convert_element_type_145 = None
        mm_36: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_108, permute_72);  view_108 = permute_72 = None
        view_109: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_36, [4, 64, 2304]);  mm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_109, 768, 2);  view_109 = None
        getitem_146: "bf16[4, 64, 768]" = split_9[0]
        getitem_147: "bf16[4, 64, 768]" = split_9[1]
        getitem_148: "bf16[4, 64, 768]" = split_9[2];  split_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_110: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_147, [4, 64, 12, 64]);  getitem_147 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_73: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_111: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_146, [4, 64, 12, 64]);  getitem_146 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_74: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_112: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_148, [4, 64, 12, 64]);  getitem_148 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_75: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_74, permute_73, permute_75, 0.0, True, scale = 0.125);  permute_74 = permute_73 = permute_75 = None
        getitem_149: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_9[0];  _scaled_dot_product_flash_attention_9 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_76: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        view_113: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_76, [4, 64, 768]);  permute_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_148: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        permute_77: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_148, [1, 0]);  convert_element_type_148 = None
        view_114: "bf16[256, 768]" = torch.ops.aten.view.default(view_113, [256, 768]);  view_113 = None
        mm_37: "bf16[256, 768]" = torch.ops.aten.mm.default(view_114, permute_77);  view_114 = permute_77 = None
        view_115: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_37, [4, 64, 768]);  mm_37 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_47: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_45, view_115);  add_45 = view_115 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_158: "f32[4, 64, 1]" = var_mean_19[0]
        getitem_159: "f32[4, 64, 1]" = var_mean_19[1];  var_mean_19 = None
        add_48: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_19: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_19: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_159);  getitem_159 = None
        mul_65: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_66: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg60_1);  mul_65 = arg60_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_151: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_152: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        permute_78: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        view_116: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_152, [256, 768]);  convert_element_type_152 = None
        mm_38: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_116, permute_78);  view_116 = permute_78 = None
        view_117: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_38, [4, 64, 3072]);  mm_38 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_155: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_67: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.5)
        mul_68: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476);  convert_element_type_155 = None
        erf_9: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_49: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_49);  mul_67 = add_49 = None
        convert_element_type_156: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_157: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_79: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_157, [1, 0]);  convert_element_type_157 = None
        view_118: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_156, [256, 3072]);  convert_element_type_156 = None
        mm_39: "bf16[256, 768]" = torch.ops.aten.mm.default(view_118, permute_79);  view_118 = permute_79 = None
        view_119: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_39, [4, 64, 768]);  mm_39 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_50: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_47, view_119);  add_47 = view_119 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_160: "f32[4, 64, 1]" = var_mean_20[0]
        getitem_161: "f32[4, 64, 1]" = var_mean_20[1];  var_mean_20 = None
        add_51: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_20: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_20: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_161);  getitem_161 = None
        mul_70: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_71: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg63_1);  mul_70 = arg63_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_160: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        convert_element_type_161: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        permute_80: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_120: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_161, [256, 768]);  convert_element_type_161 = None
        mm_40: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_120, permute_80);  view_120 = permute_80 = None
        view_121: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_40, [4, 64, 2304]);  mm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_121, 768, 2);  view_121 = None
        getitem_162: "bf16[4, 64, 768]" = split_10[0]
        getitem_163: "bf16[4, 64, 768]" = split_10[1]
        getitem_164: "bf16[4, 64, 768]" = split_10[2];  split_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_122: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_163, [4, 64, 12, 64]);  getitem_163 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_81: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_123: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_162, [4, 64, 12, 64]);  getitem_162 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_82: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_124: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_164, [4, 64, 12, 64]);  getitem_164 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_83: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_82, permute_81, permute_83, 0.0, True, scale = 0.125);  permute_82 = permute_81 = permute_83 = None
        getitem_165: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_10[0];  _scaled_dot_product_flash_attention_10 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_84: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        view_125: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_84, [4, 64, 768]);  permute_84 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_164: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        permute_85: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        view_126: "bf16[256, 768]" = torch.ops.aten.view.default(view_125, [256, 768]);  view_125 = None
        mm_41: "bf16[256, 768]" = torch.ops.aten.mm.default(view_126, permute_85);  view_126 = permute_85 = None
        view_127: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_41, [4, 64, 768]);  mm_41 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_52: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_50, view_127);  add_50 = view_127 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_174: "f32[4, 64, 1]" = var_mean_21[0]
        getitem_175: "f32[4, 64, 1]" = var_mean_21[1];  var_mean_21 = None
        add_53: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_21: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_21: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_175);  getitem_175 = None
        mul_72: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_73: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg66_1);  mul_72 = arg66_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_167: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_168: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        view_128: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_168, [256, 768]);  convert_element_type_168 = None
        mm_42: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_128, permute_86);  view_128 = permute_86 = None
        view_129: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_42, [4, 64, 3072]);  mm_42 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_171: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_74: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.5)
        mul_75: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 0.7071067811865476);  convert_element_type_171 = None
        erf_10: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_54);  mul_74 = add_54 = None
        convert_element_type_172: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_173: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_87: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        view_130: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_172, [256, 3072]);  convert_element_type_172 = None
        mm_43: "bf16[256, 768]" = torch.ops.aten.mm.default(view_130, permute_87);  view_130 = permute_87 = None
        view_131: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_43, [4, 64, 768]);  mm_43 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_55: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_52, view_131);  add_52 = view_131 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_176: "f32[4, 64, 1]" = var_mean_22[0]
        getitem_177: "f32[4, 64, 1]" = var_mean_22[1];  var_mean_22 = None
        add_56: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_22: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_22: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_177);  getitem_177 = None
        mul_77: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_78: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg69_1);  mul_77 = arg69_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:94 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        convert_element_type_176: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        convert_element_type_177: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        permute_88: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        view_132: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_177, [256, 768]);  convert_element_type_177 = None
        mm_44: "bf16[256, 2304]" = torch.ops.aten.mm.default(view_132, permute_88);  view_132 = permute_88 = None
        view_133: "bf16[4, 64, 2304]" = torch.ops.aten.view.default(mm_44, [4, 64, 2304]);  mm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_133, 768, 2);  view_133 = None
        getitem_178: "bf16[4, 64, 768]" = split_11[0]
        getitem_179: "bf16[4, 64, 768]" = split_11[1]
        getitem_180: "bf16[4, 64, 768]" = split_11[2];  split_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:95 in forward, code: k = k.view(B, T, self.n_head, C //
        view_134: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_179, [4, 64, 12, 64]);  getitem_179 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:96 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_89: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:97 in forward, code: q = q.view(B, T, self.n_head, C //
        view_135: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_178, [4, 64, 12, 64]);  getitem_178 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:98 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_90: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:99 in forward, code: v = v.view(B, T, self.n_head, C //
        view_136: "bf16[4, 64, 12, 64]" = torch.ops.aten.view.default(getitem_180, [4, 64, 12, 64]);  getitem_180 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:100 in forward, code: self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        permute_91: "bf16[4, 12, 64, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:106 in forward, code: y = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_90, permute_89, permute_91, 0.0, True, scale = 0.125);  permute_90 = permute_89 = permute_91 = None
        getitem_181: "bf16[4, 12, 64, 64]" = _scaled_dot_product_flash_attention_11[0];  _scaled_dot_product_flash_attention_11 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:118 in forward, code: y = y.transpose(1, 2).contiguous().view(B, T, C)
        permute_92: "bf16[4, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_137: "bf16[4, 64, 768]" = torch.ops.aten.view.default(permute_92, [4, 64, 768]);  permute_92 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:122 in forward, code: y = self.resid_dropout(self.c_proj(y))
        convert_element_type_180: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        permute_93: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        view_138: "bf16[256, 768]" = torch.ops.aten.view.default(view_137, [256, 768]);  view_137 = None
        mm_45: "bf16[256, 768]" = torch.ops.aten.mm.default(view_138, permute_93);  view_138 = permute_93 = None
        view_139: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_45, [4, 64, 768]);  mm_45 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:205 in forward, code: x = x + x_attn
        add_57: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_55, view_139);  add_55 = view_139 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_190: "f32[4, 64, 1]" = var_mean_23[0]
        getitem_191: "f32[4, 64, 1]" = var_mean_23[1];  var_mean_23 = None
        add_58: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_23: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_23: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_191);  getitem_191 = None
        mul_79: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_80: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_79, arg72_1);  mul_79 = arg72_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:160 in forward, code: x = self.c_fc(x)
        convert_element_type_183: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_184: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        permute_94: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_183, [1, 0]);  convert_element_type_183 = None
        view_140: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_184, [256, 768]);  convert_element_type_184 = None
        mm_46: "bf16[256, 3072]" = torch.ops.aten.mm.default(view_140, permute_94);  view_140 = permute_94 = None
        view_141: "bf16[4, 64, 3072]" = torch.ops.aten.view.default(mm_46, [4, 64, 3072]);  mm_46 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:165 in forward, code: x = self.gelu(x)
        convert_element_type_187: "f32[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_81: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5)
        mul_82: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476);  convert_element_type_187 = None
        erf_11: "f32[4, 64, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_59: "f32[4, 64, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[4, 64, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_59);  mul_81 = add_59 = None
        convert_element_type_188: "bf16[4, 64, 3072]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:174 in forward, code: x = self.c_proj(x)
        convert_element_type_189: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_95: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        view_142: "bf16[256, 3072]" = torch.ops.aten.view.default(convert_element_type_188, [256, 3072]);  convert_element_type_188 = None
        mm_47: "bf16[256, 768]" = torch.ops.aten.mm.default(view_142, permute_95);  view_142 = permute_95 = None
        view_143: "bf16[4, 64, 768]" = torch.ops.aten.view.default(mm_47, [4, 64, 768]);  mm_47 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:208 in forward, code: x = x + x_mlp
        add_60: "f32[4, 64, 768]" = torch.ops.aten.add.Tensor(add_57, view_143);  add_57 = view_143 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:49 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_192: "f32[4, 64, 1]" = var_mean_24[0]
        getitem_193: "f32[4, 64, 1]" = var_mean_24[1];  var_mean_24 = None
        add_61: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_24: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_24: "f32[4, 64, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_193);  add_60 = getitem_193 = None
        mul_84: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_85: "f32[4, 64, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg75_1);  mul_84 = arg75_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:327 in forward, code: logits = self.lm_head(x)
        convert_element_type_192: "bf16[65, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_193: "bf16[4, 64, 768]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        permute_96: "bf16[768, 65]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        view_144: "bf16[256, 768]" = torch.ops.aten.view.default(convert_element_type_193, [256, 768]);  convert_element_type_193 = None
        mm_48: "bf16[256, 65]" = torch.ops.aten.mm.default(view_144, permute_96);  view_144 = permute_96 = None
        view_145: "bf16[4, 64, 65]" = torch.ops.aten.view.default(mm_48, [4, 64, 65]);  mm_48 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:330 in forward, code: logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        view_146: "bf16[256, 65]" = torch.ops.aten.view.default(view_145, [-1, 65])
        view_147: "i64[256]" = torch.ops.aten.view.default(arg76_1, [-1]);  arg76_1 = None
        
         # File: /home/ubuntu/nanoGPT/model.py:329 in forward, code: loss = F.cross_entropy(
        convert_element_type_196: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        amax: "f32[256, 1]" = torch.ops.aten.amax.default(convert_element_type_196, [1], True)
        sub_25: "f32[256, 65]" = torch.ops.aten.sub.Tensor(convert_element_type_196, amax);  convert_element_type_196 = amax = None
        exp: "f32[256, 65]" = torch.ops.aten.exp.default(sub_25)
        sum_1: "f32[256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[256, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_26: "f32[256, 65]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_default: "f32[256, 65]" = torch.ops.prims.convert_element_type.default(sub_26, torch.float32);  sub_26 = None
        ne: "b8[256]" = torch.ops.aten.ne.Scalar(view_147, -1)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[256]" = torch.ops.aten.where.self(ne, view_147, full_default);  ne = full_default = None
        unsqueeze: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[256, 1]" = torch.ops.aten.gather.default(convert_element_type_default, 1, unsqueeze);  convert_element_type_default = unsqueeze = None
        squeeze: "f32[256]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[256]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[256]" = torch.ops.aten.ne.Scalar(view_147, -1)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[256]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2: "b8[256]" = torch.ops.aten.ne.Scalar(view_147, -1);  view_147 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_199: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_199);  sum_3 = convert_element_type_199 = None
        return (view_145, div)
        