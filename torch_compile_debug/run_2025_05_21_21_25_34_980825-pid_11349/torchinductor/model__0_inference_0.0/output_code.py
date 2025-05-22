# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_ubuntu/ln/cln62zrdezdy4hfjr6vywzofgmp2gshs673tgoqgthnb2sbjk6l3.py
# Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, layer_norm, linear], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   layer_norm => add_1, mul, mul_1, rsqrt, sub, var_mean
#   linear => convert_element_type_1
#   pos => iota
#   pos_emb => embedding_1
#   tok_emb => embedding
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %iota), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_arange_embedding_native_layer_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_arange_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_arange_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_arange_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = (xindex % 64)
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 65, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 65)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 65")
        tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
    tmp13, tmp14, tmp15 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp13[:, None]
    tmp11 = tmp14[:, None]
    tmp12 = tmp15[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 65, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 65)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 65")
        tmp21 = tl.load(in_ptr1 + (r0_2 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23 - tmp10
        tmp25 = 768.0
        tmp26 = (tmp11 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/uu/cuu7jsn46t2ga7nd6cp7inwasj37zvteg4uq2rpopfphfd2g6tyn.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/th/cth4yusycb5lp4j7p26onb6oxaqnnoi3rmi76xyyjewtrjgocgu4.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/7k/c7k47xbxwvnrxmqtphrfhm24xrlkzkvsfgmrkhf45dnluase5tuh.py
# Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, x_1, layer_norm_1, x_2], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   layer_norm_1 => add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   pos => iota
#   pos_emb => embedding_1
#   tok_emb => embedding
#   x_1 => add_2
#   x_2 => convert_element_type_8
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %iota), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_7), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_15), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_3 = async_compile.triton('triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 65), "index out of bounds: 0 <= tmp4 < 65")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [R0_BLOCK])
    tmp26 = tl.where(r0_mask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp11 - tmp21
    tmp29 = 768.0
    tmp30 = (tmp27 / tmp29)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp37, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/ri/cripiff4h6eku2ifipfslmy4hmdasegzt37kj5jw7ckyv2e5sxrz.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/2p/c2p2l4z4yt6eg3bxo2odkwfcfc74anv4ebyjmkvurfc2mqxjakbp.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_3 => add_4, convert_element_type_11, convert_element_type_12, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %convert_element_type_11 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_4), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_5 = async_compile.triton('triton_poi_fused_gelu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/7v/c7vg3bpxde3kcjpnocjoda5jprwvfn3qht735hx33rv6byv6of6e.py
# Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, x_1, x_6, layer_norm_2, linear_4], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   layer_norm_2 => add_6, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
#   linear_4 => convert_element_type_17
#   pos => iota
#   pos_emb => embedding_1
#   tok_emb => embedding
#   x_1 => add_2
#   x_6 => add_5
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %iota), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_7), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_11), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_17), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %arg9_1), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_6 = async_compile.triton('triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 65), "index out of bounds: 0 <= tmp4 < 65")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 768, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp14 - tmp24
    tmp32 = 768.0
    tmp33 = (tmp30 / tmp32)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp14, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/yt/cytcmelccqikmrqwgxqrkma7arfchbavdlpfnac7vrnubqfjnkhe.py
# Topologically Sorted Source Nodes: [x_7, layer_norm_3, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm_3 => add_8, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_7 => add_7
#   x_8 => convert_element_type_24
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_19), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_31), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg12_1), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_7 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 768.0
    tmp22 = (tmp19 / tmp21)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/f7/cf7i7blsahanh5bpr4ztcoy4ycfpx4mgum6tvnguczlgjxvsgt6d.py
# Topologically Sorted Source Nodes: [x_7, x_12, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm_4 => add_11, mul_14, mul_15, rsqrt_4, sub_4, var_mean_4
#   linear_8 => convert_element_type_33
#   x_12 => add_10
#   x_7 => add_7
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_19), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_33), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %arg15_1), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_8 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = (tmp22 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/gt/cgtrz4c3riw4rw7zjh3nmx26cx435s3vvo5thtbnbtv43liz6va2.py
# Topologically Sorted Source Nodes: [x_7, x_12, x_13, layer_norm_5, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm_5 => add_13, mul_16, mul_17, rsqrt_5, sub_5, var_mean_5
#   x_12 => add_10
#   x_13 => add_12
#   x_14 => convert_element_type_40
#   x_7 => add_7
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_19), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_31), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_47), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg18_1), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_9 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 768.0
    tmp28 = (tmp25 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/fd/cfdtqwtc6bibryqws3es72ddcju6bxkwz2onunlzqxueya6avhkm.py
# Topologically Sorted Source Nodes: [x_7, x_12, x_13, x_18, layer_norm_6, linear_12], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm_6 => add_16, mul_21, mul_22, rsqrt_6, sub_6, var_mean_6
#   linear_12 => convert_element_type_49
#   x_12 => add_10
#   x_13 => add_12
#   x_18 => add_15
#   x_7 => add_7
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_19), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_31), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_35), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_49), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %arg21_1), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = (tmp28 / tmp30)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp38, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/up/cupzyoidfucmyvbtq6jxs2p7g7pfiws6ja6rp27j5y3doyoargdk.py
# Topologically Sorted Source Nodes: [x_67, x_72, x_73, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   logits => convert_element_type_193
#   x_67 => add_57
#   x_72 => add_60
#   x_73 => add_61, mul_84, mul_85, rsqrt_24, sub_24, var_mean_24
# Graph fragment:
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %view_139), kwargs = {})
#   %add_60 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %view_143), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_60, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_60, %getitem_193), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_61,), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %arg75_1), kwargs = {})
#   %convert_element_type_193 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_85, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_11 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = (tmp22 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp32, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/hu/chunwqszh5x6rwcy6wrimldyuu2wixot3s546n7np5zoqlnxna7u.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   logits => convert_element_type_192
# Graph fragment:
#   %convert_element_type_192 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/wu/cwuy3pghvpw2kell7l5epbofhc7lp4idzxsae2kea3ekxnlr6qtl.py
# Topologically Sorted Source Nodes: [loss, ], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   loss => convert_element_type_196
# Graph fragment:
#   %convert_element_type_196 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_146, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_196, 1), kwargs = {})
triton_per_fused__log_softmax_prepare_softmax_online_13 = async_compile.triton('triton_per_fused__log_softmax_prepare_softmax_online_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_prepare_softmax_online_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_prepare_softmax_online_13(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 65
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 65*x0), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp2 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/p5/cp5fdkwvp6usr6uhr6e4qww26jh7lv4ilkqfwhikeg5pzj5eb4dc.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_199, div, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_147, -1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_147, -1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_199 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_199), kwargs = {})
triton_per_fused_nll_loss_forward_14 = async_compile.triton('triton_per_fused_nll_loss_forward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_nll_loss_forward_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp12 = tl.load(in_ptr2 + (r0_0), None)
    tmp14 = tl.load(in_ptr3 + (r0_0), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < 65), "index out of bounds: 0 <= tmp8 < 65")
    tmp10 = tl.load(in_ptr1 + (tmp8 + 65*r0_0), None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tl_math.log(tmp14)
    tmp16 = tmp13 - tmp15
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp2, tmp17, tmp18)
    tmp20 = tl.broadcast_to(tmp19, [R0_BLOCK])
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp2.to(tl.int64)
    tmp24 = tl.broadcast_to(tmp23, [R0_BLOCK])
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = tmp26.to(tl.float32)
    tmp28 = (tmp22 / tmp27)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 64), (64, 1))
    assert_size_stride(arg1_1, (65, 768), (768, 1))
    assert_size_stride(arg2_1, (64, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (2304, 768), (768, 1))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (3072, 768), (768, 1))
    assert_size_stride(arg8_1, (768, 3072), (3072, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (2304, 768), (768, 1))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (3072, 768), (768, 1))
    assert_size_stride(arg14_1, (768, 3072), (3072, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (2304, 768), (768, 1))
    assert_size_stride(arg17_1, (768, 768), (768, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (3072, 768), (768, 1))
    assert_size_stride(arg20_1, (768, 3072), (3072, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (2304, 768), (768, 1))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (3072, 768), (768, 1))
    assert_size_stride(arg26_1, (768, 3072), (3072, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (2304, 768), (768, 1))
    assert_size_stride(arg29_1, (768, 768), (768, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (768, 3072), (3072, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (2304, 768), (768, 1))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (3072, 768), (768, 1))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (2304, 768), (768, 1))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (3072, 768), (768, 1))
    assert_size_stride(arg44_1, (768, 3072), (3072, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (2304, 768), (768, 1))
    assert_size_stride(arg47_1, (768, 768), (768, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (3072, 768), (768, 1))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (2304, 768), (768, 1))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (3072, 768), (768, 1))
    assert_size_stride(arg56_1, (768, 3072), (3072, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (2304, 768), (768, 1))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (3072, 768), (768, 1))
    assert_size_stride(arg62_1, (768, 3072), (3072, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (2304, 768), (768, 1))
    assert_size_stride(arg65_1, (768, 768), (768, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (3072, 768), (768, 1))
    assert_size_stride(arg68_1, (768, 3072), (3072, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (2304, 768), (768, 1))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (3072, 768), (768, 1))
    assert_size_stride(arg74_1, (768, 3072), (3072, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (4, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, layer_norm, linear], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_arange_embedding_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, arg3_1, buf3, 256, 768, stream=stream0)
        del arg3_1
        buf4 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg4_1, buf4, 1769472, stream=stream0)
        del arg4_1
        buf5 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (256, 768), (768, 1), 0), reinterpret_tensor(buf4, (768, 2304), (1, 768), 0), out=buf5)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf6 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf5, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf5, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf5, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf7 = buf6[0]
        assert_size_stride(buf7, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf6
        buf12 = reinterpret_tensor(buf5, (768, 768), (768, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg5_1, buf12, 589824, stream=stream0)
        del arg5_1
        buf13 = reinterpret_tensor(buf3, (256, 768), (768, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (256, 768), (768, 1), 0), reinterpret_tensor(buf12, (768, 768), (1, 768), 0), out=buf13)
        buf17 = reinterpret_tensor(buf7, (4, 64, 768), (49152, 768, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, x_1, layer_norm_1, x_2], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_3.run(arg0_1, arg1_1, arg2_1, buf13, arg6_1, buf17, 256, 768, stream=stream0)
        del arg6_1
        buf18 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg7_1, buf18, 2359296, stream=stream0)
        del arg7_1
        buf19 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (256, 768), (768, 1), 0), reinterpret_tensor(buf18, (768, 3072), (1, 768), 0), out=buf19)
        buf20 = reinterpret_tensor(buf19, (4, 64, 3072), (196608, 3072, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf20, 786432, stream=stream0)
        buf21 = reinterpret_tensor(buf18, (768, 3072), (3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg8_1, buf21, 2359296, stream=stream0)
        del arg8_1
        buf22 = reinterpret_tensor(buf17, (256, 768), (768, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf21, (3072, 768), (1, 3072), 0), out=buf22)
        buf23 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf27 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tok_emb, pos, pos_emb, add, x_1, x_6, layer_norm_2, linear_4], Original ATen: [aten.embedding, aten.arange, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_arange_embedding_native_layer_norm_6.run(arg0_1, arg1_1, arg2_1, buf13, buf22, arg9_1, buf23, buf27, 256, 768, stream=stream0)
        del arg0_1
        del arg2_1
        del arg9_1
        buf28 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg10_1, buf28, 1769472, stream=stream0)
        del arg10_1
        buf29 = reinterpret_tensor(buf12, (256, 2304), (2304, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (256, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, 2304), (1, 768), 0), out=buf29)
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf30 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf29, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf29, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf29, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf31 = buf30[0]
        assert_size_stride(buf31, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf30
        buf36 = reinterpret_tensor(buf29, (768, 768), (768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg11_1, buf36, 589824, stream=stream0)
        del arg11_1
        buf37 = reinterpret_tensor(buf27, (256, 768), (768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (256, 768), (768, 1), 0), reinterpret_tensor(buf36, (768, 768), (1, 768), 0), out=buf37)
        buf41 = reinterpret_tensor(buf31, (4, 64, 768), (49152, 768, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_7, layer_norm_3, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf23, buf37, arg12_1, buf41, 256, 768, stream=stream0)
        del arg12_1
        buf42 = reinterpret_tensor(buf21, (3072, 768), (768, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg13_1, buf42, 2359296, stream=stream0)
        del arg13_1
        buf43 = reinterpret_tensor(buf20, (256, 3072), (3072, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (256, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, 3072), (1, 768), 0), out=buf43)
        buf44 = reinterpret_tensor(buf43, (4, 64, 3072), (196608, 3072, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf44, 786432, stream=stream0)
        buf45 = reinterpret_tensor(buf42, (768, 3072), (3072, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg14_1, buf45, 2359296, stream=stream0)
        del arg14_1
        buf46 = reinterpret_tensor(buf41, (256, 768), (768, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf45, (3072, 768), (1, 3072), 0), out=buf46)
        buf50 = reinterpret_tensor(buf22, (4, 64, 768), (49152, 768, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_12, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_8.run(buf23, buf37, buf46, arg15_1, buf50, 256, 768, stream=stream0)
        del arg15_1
        buf51 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg16_1, buf51, 1769472, stream=stream0)
        del arg16_1
        buf52 = reinterpret_tensor(buf36, (256, 2304), (2304, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (256, 768), (768, 1), 0), reinterpret_tensor(buf51, (768, 2304), (1, 768), 0), out=buf52)
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf53 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf52, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf52, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf52, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf54 = buf53[0]
        assert_size_stride(buf54, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf53
        buf59 = reinterpret_tensor(buf52, (768, 768), (768, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg17_1, buf59, 589824, stream=stream0)
        del arg17_1
        buf60 = reinterpret_tensor(buf50, (256, 768), (768, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (256, 768), (768, 1), 0), reinterpret_tensor(buf59, (768, 768), (1, 768), 0), out=buf60)
        buf64 = reinterpret_tensor(buf54, (4, 64, 768), (49152, 768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_12, x_13, layer_norm_5, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_9.run(buf23, buf37, buf46, buf60, arg18_1, buf64, 256, 768, stream=stream0)
        del arg18_1
        buf65 = reinterpret_tensor(buf45, (3072, 768), (768, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg19_1, buf65, 2359296, stream=stream0)
        del arg19_1
        buf66 = reinterpret_tensor(buf44, (256, 3072), (3072, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (256, 768), (768, 1), 0), reinterpret_tensor(buf65, (768, 3072), (1, 768), 0), out=buf66)
        buf67 = reinterpret_tensor(buf66, (4, 64, 3072), (196608, 3072, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf67, 786432, stream=stream0)
        buf68 = reinterpret_tensor(buf65, (768, 3072), (3072, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg20_1, buf68, 2359296, stream=stream0)
        del arg20_1
        buf69 = reinterpret_tensor(buf64, (256, 768), (768, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf68, (3072, 768), (1, 3072), 0), out=buf69)
        buf70 = buf23; del buf23  # reuse
        buf74 = reinterpret_tensor(buf13, (4, 64, 768), (49152, 768, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_12, x_13, x_18, layer_norm_6, linear_12], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_10.run(buf70, buf37, buf46, buf60, buf69, arg21_1, buf74, 256, 768, stream=stream0)
        del arg21_1
        del buf37
        del buf46
        buf75 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg22_1, buf75, 1769472, stream=stream0)
        del arg22_1
        buf76 = reinterpret_tensor(buf59, (256, 2304), (2304, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (256, 768), (768, 1), 0), reinterpret_tensor(buf75, (768, 2304), (1, 768), 0), out=buf76)
        # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf77 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf76, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf76, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf76, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf78 = buf77[0]
        assert_size_stride(buf78, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf77
        buf83 = reinterpret_tensor(buf76, (768, 768), (768, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg23_1, buf83, 589824, stream=stream0)
        del arg23_1
        buf84 = reinterpret_tensor(buf74, (256, 768), (768, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (256, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, 768), (1, 768), 0), out=buf84)
        buf88 = reinterpret_tensor(buf78, (4, 64, 768), (49152, 768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_19, layer_norm_7, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf70, buf84, arg24_1, buf88, 256, 768, stream=stream0)
        del arg24_1
        buf89 = reinterpret_tensor(buf68, (3072, 768), (768, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg25_1, buf89, 2359296, stream=stream0)
        del arg25_1
        buf90 = reinterpret_tensor(buf67, (256, 3072), (3072, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (256, 768), (768, 1), 0), reinterpret_tensor(buf89, (768, 3072), (1, 768), 0), out=buf90)
        buf91 = reinterpret_tensor(buf90, (4, 64, 3072), (196608, 3072, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf91, 786432, stream=stream0)
        buf92 = reinterpret_tensor(buf89, (768, 3072), (3072, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg26_1, buf92, 2359296, stream=stream0)
        del arg26_1
        buf93 = reinterpret_tensor(buf88, (256, 768), (768, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf92, (3072, 768), (1, 3072), 0), out=buf93)
        buf97 = reinterpret_tensor(buf69, (4, 64, 768), (49152, 768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_19, x_24, layer_norm_8, linear_16], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_8.run(buf70, buf84, buf93, arg27_1, buf97, 256, 768, stream=stream0)
        del arg27_1
        buf98 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg28_1, buf98, 1769472, stream=stream0)
        del arg28_1
        buf99 = reinterpret_tensor(buf83, (256, 2304), (2304, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (256, 768), (768, 1), 0), reinterpret_tensor(buf98, (768, 2304), (1, 768), 0), out=buf99)
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf100 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf99, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf101 = buf100[0]
        assert_size_stride(buf101, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf100
        buf106 = reinterpret_tensor(buf99, (768, 768), (768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg29_1, buf106, 589824, stream=stream0)
        del arg29_1
        buf107 = reinterpret_tensor(buf97, (256, 768), (768, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (256, 768), (768, 1), 0), reinterpret_tensor(buf106, (768, 768), (1, 768), 0), out=buf107)
        buf111 = reinterpret_tensor(buf101, (4, 64, 768), (49152, 768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_19, x_24, x_25, layer_norm_9, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_9.run(buf70, buf84, buf93, buf107, arg30_1, buf111, 256, 768, stream=stream0)
        del arg30_1
        buf112 = reinterpret_tensor(buf92, (3072, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg31_1, buf112, 2359296, stream=stream0)
        del arg31_1
        buf113 = reinterpret_tensor(buf91, (256, 3072), (3072, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (256, 768), (768, 1), 0), reinterpret_tensor(buf112, (768, 3072), (1, 768), 0), out=buf113)
        buf114 = reinterpret_tensor(buf113, (4, 64, 3072), (196608, 3072, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf114, 786432, stream=stream0)
        buf115 = reinterpret_tensor(buf112, (768, 3072), (3072, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg32_1, buf115, 2359296, stream=stream0)
        del arg32_1
        buf116 = reinterpret_tensor(buf111, (256, 768), (768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf114, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf115, (3072, 768), (1, 3072), 0), out=buf116)
        buf117 = buf70; del buf70  # reuse
        buf121 = reinterpret_tensor(buf60, (4, 64, 768), (49152, 768, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_19, x_24, x_25, x_30, layer_norm_10, linear_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_10.run(buf117, buf84, buf93, buf107, buf116, arg33_1, buf121, 256, 768, stream=stream0)
        del arg33_1
        del buf107
        del buf116
        buf122 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg34_1, buf122, 1769472, stream=stream0)
        del arg34_1
        buf123 = reinterpret_tensor(buf106, (256, 2304), (2304, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (256, 768), (768, 1), 0), reinterpret_tensor(buf122, (768, 2304), (1, 768), 0), out=buf123)
        # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf124 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf123, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf123, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf123, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf125 = buf124[0]
        assert_size_stride(buf125, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf124
        buf130 = reinterpret_tensor(buf123, (768, 768), (768, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg35_1, buf130, 589824, stream=stream0)
        del arg35_1
        buf131 = reinterpret_tensor(buf121, (256, 768), (768, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (256, 768), (768, 1), 0), reinterpret_tensor(buf130, (768, 768), (1, 768), 0), out=buf131)
        buf135 = reinterpret_tensor(buf125, (4, 64, 768), (49152, 768, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_31, layer_norm_11, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf117, buf131, arg36_1, buf135, 256, 768, stream=stream0)
        del arg36_1
        buf136 = reinterpret_tensor(buf115, (3072, 768), (768, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg37_1, buf136, 2359296, stream=stream0)
        del arg37_1
        buf137 = reinterpret_tensor(buf114, (256, 3072), (3072, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (256, 768), (768, 1), 0), reinterpret_tensor(buf136, (768, 3072), (1, 768), 0), out=buf137)
        buf138 = reinterpret_tensor(buf137, (4, 64, 3072), (196608, 3072, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf138, 786432, stream=stream0)
        buf139 = reinterpret_tensor(buf136, (768, 3072), (3072, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg38_1, buf139, 2359296, stream=stream0)
        del arg38_1
        buf140 = reinterpret_tensor(buf135, (256, 768), (768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf139, (3072, 768), (1, 3072), 0), out=buf140)
        buf144 = reinterpret_tensor(buf93, (4, 64, 768), (49152, 768, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_31, x_36, layer_norm_12, linear_24], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_8.run(buf117, buf131, buf140, arg39_1, buf144, 256, 768, stream=stream0)
        del arg39_1
        buf145 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg40_1, buf145, 1769472, stream=stream0)
        del arg40_1
        buf146 = reinterpret_tensor(buf130, (256, 2304), (2304, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (256, 768), (768, 1), 0), reinterpret_tensor(buf145, (768, 2304), (1, 768), 0), out=buf146)
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf147 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf146, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf146, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf146, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf148 = buf147[0]
        assert_size_stride(buf148, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf147
        buf153 = reinterpret_tensor(buf146, (768, 768), (768, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg41_1, buf153, 589824, stream=stream0)
        del arg41_1
        buf154 = reinterpret_tensor(buf144, (256, 768), (768, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (256, 768), (768, 1), 0), reinterpret_tensor(buf153, (768, 768), (1, 768), 0), out=buf154)
        buf158 = reinterpret_tensor(buf148, (4, 64, 768), (49152, 768, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_31, x_36, x_37, layer_norm_13, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_9.run(buf117, buf131, buf140, buf154, arg42_1, buf158, 256, 768, stream=stream0)
        del arg42_1
        buf159 = reinterpret_tensor(buf139, (3072, 768), (768, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg43_1, buf159, 2359296, stream=stream0)
        del arg43_1
        buf160 = reinterpret_tensor(buf138, (256, 3072), (3072, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (256, 768), (768, 1), 0), reinterpret_tensor(buf159, (768, 3072), (1, 768), 0), out=buf160)
        buf161 = reinterpret_tensor(buf160, (4, 64, 3072), (196608, 3072, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf161, 786432, stream=stream0)
        buf162 = reinterpret_tensor(buf159, (768, 3072), (3072, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg44_1, buf162, 2359296, stream=stream0)
        del arg44_1
        buf163 = reinterpret_tensor(buf158, (256, 768), (768, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf162, (3072, 768), (1, 3072), 0), out=buf163)
        buf164 = buf117; del buf117  # reuse
        buf168 = reinterpret_tensor(buf84, (4, 64, 768), (49152, 768, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_31, x_36, x_37, x_42, layer_norm_14, linear_28], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_10.run(buf164, buf131, buf140, buf154, buf163, arg45_1, buf168, 256, 768, stream=stream0)
        del arg45_1
        del buf131
        del buf140
        buf169 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg46_1, buf169, 1769472, stream=stream0)
        del arg46_1
        buf170 = reinterpret_tensor(buf153, (256, 2304), (2304, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (256, 768), (768, 1), 0), reinterpret_tensor(buf169, (768, 2304), (1, 768), 0), out=buf170)
        # Topologically Sorted Source Nodes: [y_21], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf171 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf170, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf170, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf170, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf172 = buf171[0]
        assert_size_stride(buf172, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf171
        buf177 = reinterpret_tensor(buf170, (768, 768), (768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg47_1, buf177, 589824, stream=stream0)
        del arg47_1
        buf178 = reinterpret_tensor(buf168, (256, 768), (768, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (256, 768), (768, 1), 0), reinterpret_tensor(buf177, (768, 768), (1, 768), 0), out=buf178)
        buf182 = reinterpret_tensor(buf172, (4, 64, 768), (49152, 768, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_43, layer_norm_15, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf164, buf178, arg48_1, buf182, 256, 768, stream=stream0)
        del arg48_1
        buf183 = reinterpret_tensor(buf162, (3072, 768), (768, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg49_1, buf183, 2359296, stream=stream0)
        del arg49_1
        buf184 = reinterpret_tensor(buf161, (256, 3072), (3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (256, 768), (768, 1), 0), reinterpret_tensor(buf183, (768, 3072), (1, 768), 0), out=buf184)
        buf185 = reinterpret_tensor(buf184, (4, 64, 3072), (196608, 3072, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf185, 786432, stream=stream0)
        buf186 = reinterpret_tensor(buf183, (768, 3072), (3072, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg50_1, buf186, 2359296, stream=stream0)
        del arg50_1
        buf187 = reinterpret_tensor(buf182, (256, 768), (768, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf186, (3072, 768), (1, 3072), 0), out=buf187)
        buf191 = reinterpret_tensor(buf163, (4, 64, 768), (49152, 768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_43, x_48, layer_norm_16, linear_32], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_8.run(buf164, buf178, buf187, arg51_1, buf191, 256, 768, stream=stream0)
        del arg51_1
        buf192 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg52_1, buf192, 1769472, stream=stream0)
        del arg52_1
        buf193 = reinterpret_tensor(buf177, (256, 2304), (2304, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (256, 768), (768, 1), 0), reinterpret_tensor(buf192, (768, 2304), (1, 768), 0), out=buf193)
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf194 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf193, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf193, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf193, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf195 = buf194[0]
        assert_size_stride(buf195, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf194
        buf200 = reinterpret_tensor(buf193, (768, 768), (768, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg53_1, buf200, 589824, stream=stream0)
        del arg53_1
        buf201 = reinterpret_tensor(buf191, (256, 768), (768, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (256, 768), (768, 1), 0), reinterpret_tensor(buf200, (768, 768), (1, 768), 0), out=buf201)
        buf205 = reinterpret_tensor(buf195, (4, 64, 768), (49152, 768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_43, x_48, x_49, layer_norm_17, x_50], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_9.run(buf164, buf178, buf187, buf201, arg54_1, buf205, 256, 768, stream=stream0)
        del arg54_1
        buf206 = reinterpret_tensor(buf186, (3072, 768), (768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg55_1, buf206, 2359296, stream=stream0)
        del arg55_1
        buf207 = reinterpret_tensor(buf185, (256, 3072), (3072, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (256, 768), (768, 1), 0), reinterpret_tensor(buf206, (768, 3072), (1, 768), 0), out=buf207)
        buf208 = reinterpret_tensor(buf207, (4, 64, 3072), (196608, 3072, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf208, 786432, stream=stream0)
        buf209 = reinterpret_tensor(buf206, (768, 3072), (3072, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg56_1, buf209, 2359296, stream=stream0)
        del arg56_1
        buf210 = reinterpret_tensor(buf205, (256, 768), (768, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf209, (3072, 768), (1, 3072), 0), out=buf210)
        buf211 = buf164; del buf164  # reuse
        buf215 = reinterpret_tensor(buf154, (4, 64, 768), (49152, 768, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_43, x_48, x_49, x_54, layer_norm_18, linear_36], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_10.run(buf211, buf178, buf187, buf201, buf210, arg57_1, buf215, 256, 768, stream=stream0)
        del arg57_1
        del buf178
        del buf187
        buf216 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg58_1, buf216, 1769472, stream=stream0)
        del arg58_1
        buf217 = reinterpret_tensor(buf200, (256, 2304), (2304, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (256, 768), (768, 1), 0), reinterpret_tensor(buf216, (768, 2304), (1, 768), 0), out=buf217)
        # Topologically Sorted Source Nodes: [y_27], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf218 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf217, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf217, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf217, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf219 = buf218[0]
        assert_size_stride(buf219, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf218
        buf224 = reinterpret_tensor(buf217, (768, 768), (768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg59_1, buf224, 589824, stream=stream0)
        del arg59_1
        buf225 = reinterpret_tensor(buf215, (256, 768), (768, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (256, 768), (768, 1), 0), reinterpret_tensor(buf224, (768, 768), (1, 768), 0), out=buf225)
        buf229 = reinterpret_tensor(buf219, (4, 64, 768), (49152, 768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_55, layer_norm_19, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf211, buf225, arg60_1, buf229, 256, 768, stream=stream0)
        del arg60_1
        buf230 = reinterpret_tensor(buf209, (3072, 768), (768, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg61_1, buf230, 2359296, stream=stream0)
        del arg61_1
        buf231 = reinterpret_tensor(buf208, (256, 3072), (3072, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (256, 768), (768, 1), 0), reinterpret_tensor(buf230, (768, 3072), (1, 768), 0), out=buf231)
        buf232 = reinterpret_tensor(buf231, (4, 64, 3072), (196608, 3072, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf232, 786432, stream=stream0)
        buf233 = reinterpret_tensor(buf230, (768, 3072), (3072, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg62_1, buf233, 2359296, stream=stream0)
        del arg62_1
        buf234 = reinterpret_tensor(buf229, (256, 768), (768, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf233, (3072, 768), (1, 3072), 0), out=buf234)
        buf238 = reinterpret_tensor(buf210, (4, 64, 768), (49152, 768, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_55, x_60, layer_norm_20, linear_40], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_8.run(buf211, buf225, buf234, arg63_1, buf238, 256, 768, stream=stream0)
        del arg63_1
        buf239 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg64_1, buf239, 1769472, stream=stream0)
        del arg64_1
        buf240 = reinterpret_tensor(buf224, (256, 2304), (2304, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (256, 768), (768, 1), 0), reinterpret_tensor(buf239, (768, 2304), (1, 768), 0), out=buf240)
        # Topologically Sorted Source Nodes: [y_30], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf241 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf240, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf240, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf240, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf242 = buf241[0]
        assert_size_stride(buf242, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf241
        buf247 = reinterpret_tensor(buf240, (768, 768), (768, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg65_1, buf247, 589824, stream=stream0)
        del arg65_1
        buf248 = reinterpret_tensor(buf238, (256, 768), (768, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (256, 768), (768, 1), 0), reinterpret_tensor(buf247, (768, 768), (1, 768), 0), out=buf248)
        buf252 = reinterpret_tensor(buf242, (4, 64, 768), (49152, 768, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_55, x_60, x_61, layer_norm_21, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_9.run(buf211, buf225, buf234, buf248, arg66_1, buf252, 256, 768, stream=stream0)
        del arg66_1
        buf253 = reinterpret_tensor(buf233, (3072, 768), (768, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg67_1, buf253, 2359296, stream=stream0)
        del arg67_1
        buf254 = reinterpret_tensor(buf232, (256, 3072), (3072, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (256, 768), (768, 1), 0), reinterpret_tensor(buf253, (768, 3072), (1, 768), 0), out=buf254)
        buf255 = reinterpret_tensor(buf254, (4, 64, 3072), (196608, 3072, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf255, 786432, stream=stream0)
        buf256 = reinterpret_tensor(buf253, (768, 3072), (3072, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg68_1, buf256, 2359296, stream=stream0)
        del arg68_1
        buf257 = reinterpret_tensor(buf252, (256, 768), (768, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf256, (3072, 768), (1, 3072), 0), out=buf257)
        buf258 = buf211; del buf211  # reuse
        buf262 = reinterpret_tensor(buf201, (4, 64, 768), (49152, 768, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_55, x_60, x_61, x_66, layer_norm_22, linear_44], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_10.run(buf258, buf225, buf234, buf248, buf257, arg69_1, buf262, 256, 768, stream=stream0)
        del arg69_1
        del buf225
        del buf234
        del buf248
        del buf257
        buf263 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg70_1, buf263, 1769472, stream=stream0)
        del arg70_1
        buf264 = reinterpret_tensor(buf247, (256, 2304), (2304, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (256, 768), (768, 1), 0), reinterpret_tensor(buf263, (768, 2304), (1, 768), 0), out=buf264)
        del buf263
        # Topologically Sorted Source Nodes: [y_33], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf265 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf266 = buf265[0]
        assert_size_stride(buf266, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf265
        buf271 = reinterpret_tensor(buf264, (768, 768), (768, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg71_1, buf271, 589824, stream=stream0)
        del arg71_1
        buf272 = reinterpret_tensor(buf262, (256, 768), (768, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (256, 768), (768, 1), 0), reinterpret_tensor(buf271, (768, 768), (1, 768), 0), out=buf272)
        del buf271
        buf276 = reinterpret_tensor(buf266, (4, 64, 768), (49152, 768, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_67, layer_norm_23, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_7.run(buf258, buf272, arg72_1, buf276, 256, 768, stream=stream0)
        del arg72_1
        buf277 = reinterpret_tensor(buf256, (3072, 768), (768, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg73_1, buf277, 2359296, stream=stream0)
        del arg73_1
        buf278 = reinterpret_tensor(buf255, (256, 3072), (3072, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (256, 768), (768, 1), 0), reinterpret_tensor(buf277, (768, 3072), (1, 768), 0), out=buf278)
        buf279 = reinterpret_tensor(buf278, (4, 64, 3072), (196608, 3072, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf279, 786432, stream=stream0)
        buf280 = reinterpret_tensor(buf277, (768, 3072), (3072, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg74_1, buf280, 2359296, stream=stream0)
        del arg74_1
        buf281 = reinterpret_tensor(buf276, (256, 768), (768, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf280, (3072, 768), (1, 3072), 0), out=buf281)
        del buf279
        del buf280
        buf285 = reinterpret_tensor(buf272, (4, 64, 768), (49152, 768, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_72, x_73, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_11.run(buf285, buf258, buf281, arg75_1, 256, 768, stream=stream0)
        del arg75_1
        del buf258
        del buf281
        buf286 = empty_strided_cuda((65, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(arg1_1, buf286, 49920, stream=stream0)
        del arg1_1
        buf287 = empty_strided_cuda((256, 65), (65, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (256, 768), (768, 1), 0), reinterpret_tensor(buf286, (768, 65), (1, 768), 0), out=buf287)
        del buf285
        del buf286
        buf288 = empty_strided_cuda((256, 1), (1, 256), torch.float32)
        buf289 = empty_strided_cuda((256, 1), (1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [loss, ], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_prepare_softmax_online_13.run(buf287, buf288, buf289, 256, 65, stream=stream0)
        buf290 = empty_strided_cuda((), (), torch.float32)
        buf292 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_per_fused_nll_loss_forward_14.run(buf292, arg76_1, buf287, buf288, buf289, 1, 256, stream=stream0)
        del arg76_1
        del buf288
        del buf289
    return (reinterpret_tensor(buf287, (4, 64, 65), (4160, 65, 1), 0), buf292, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((65, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
