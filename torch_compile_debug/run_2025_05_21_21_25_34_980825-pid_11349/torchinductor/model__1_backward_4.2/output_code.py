# AOT ID: ['1_backward']
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


# kernel path: /tmp/torchinductor_ubuntu/gb/cgbsznafz5gqdpjntgattl64bftk25nlio4qq2kcafrz2t7kiraj.py
# Topologically Sorted Source Nodes: [loss, scatter, convert_element_type_default], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data, aten.add]
# Source node to ATen node mapping:
#   convert_element_type_default => mul_86
#   loss => full_default, full_default_1
#   scatter => scatter_upon_const_tensor
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_2, %convert_element_type_199), kwargs = {})
#   %ne_3 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [256, 65], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_1, %full_default_1), kwargs = {})
#   %mul_86 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_86, [1], True), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %view_148), kwargs = {})
triton_per_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_per_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp21 = tl.load(in_ptr3 + (r0_1 + 65*x0), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (r0_1 + 65*x0), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], -1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = r0_1
    tmp6 = tmp4 == tmp5
    tmp7 = -1.0
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tl.where(tmp2, tmp14, tmp8)
    tmp16 = tmp9 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp30 * tmp20
    tmp32 = tmp16 - tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp21 + tmp33
    tl.store(out_ptr1 + (r0_1 + 65*x0), tmp34, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/kj/ckjfh7vbrps5sdkyfwgctgqngm4xbh7zijt62ofr3jyutjd2lmua.py
# Topologically Sorted Source Nodes: [full_1], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_1 => full_default_6
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_1 = async_compile.triton('triton_poi_fused_embedding_dense_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/ap/cap7xwklk4bvdt4ec7rqmawmrpjjaobersloo2oembn4tslvwpnc.py
# Topologically Sorted Source Nodes: [full_2], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_2 => full_default_8
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([65, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_2 = async_compile.triton('triton_poi_fused_embedding_dense_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/tt/ctttacyfkbw7dvpc4j6b6vfcjmiqwlwz54na4xcqzkddfeflkwxf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_208 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_150, torch.float32), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_208, %mul_84), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_94, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_3 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 98304*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*r0_2 + 98304*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/e4/ce4ijbylkfybqctkyknwf235yohnvdp7z3q46qktly3l7fofclbl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_208 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_150, torch.float32), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_208, %mul_84), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_94, [0, 1]), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_backward_4 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_backward_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_backward_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_backward_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/g4/cg42wypsazmbcy4jnrktfehnkskqjwle4d3vikoivjoz2kpalwbe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_208 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_150, torch.float32), kwargs = {})
#   %mul_89 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_208, %primals_76), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, 768), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_89, [2], True), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %mul_84), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_91, [2], True), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %sum_6), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_90, %sum_5), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_29, %mul_92), kwargs = {})
#   %mul_93 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_30), kwargs = {})
#   %convert_element_type_210 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.bfloat16), kwargs = {})
#   %view_151 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_210, [256, 768]), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_backward_view_5 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_backward_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_backward_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_backward_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 768.0
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 - tmp7
    tmp18 = tmp8 * tmp13
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/wx/cwxdgdxgiemodi6kmwdadbr4ipcwmjynw6pnxjfehgxorl67ui6w.py
# Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   x_69 => add_59, convert_element_type_187, erf_11, mul_82
# Graph fragment:
#   %convert_element_type_216 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_152, torch.float32), kwargs = {})
#   %convert_element_type_187 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_141, torch.float32), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_187, 0.7071067811865476), kwargs = {})
#   %erf_11 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_82,), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_11, 1), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, 0.5), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_187, %convert_element_type_187), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, -0.5), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_98,), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, 0.3989422804014327), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_187, %mul_99), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %mul_100), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_216, %add_64), kwargs = {})
#   %convert_element_type_218 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_101, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/jp/cjpfakvx5z7uwp6c2kxbj4gy2pyhisu7hmeeo5fucqo4wsrypbld.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_223 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_154, torch.float32), kwargs = {})
#   %mul_103 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_223, %primals_73), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, 768), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_103, [2], True), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %mul_79), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_105, [2], True), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %sum_9), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_104, %sum_8), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %mul_106), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_33), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %mul_107), kwargs = {})
#   %convert_element_type_225 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_65, torch.bfloat16), kwargs = {})
#   %view_155 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_225, [256, 768]), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_backward_view_7 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_backward_view_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_backward_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_backward_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp14 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = 768.0
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 - tmp7
    tmp19 = tmp8 * tmp13
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp14 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp23, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/a7/ca75wy3r3aue7vt2syfw7t7qecdqqhkpfhezvnh5vhhobvo2g5xg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
# Source node to ATen node mapping:
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_159, %view_160, %view_158], 2), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2304)
    x1 = xindex // 2304
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (768*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (768*x1 + ((-768) + x0)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 2304, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (768*x1 + ((-1536) + x0)), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/cq/ccqmurhr35ss3y5u5shoxbmv4ycdwkkmryox3qeyny6t7ox4p25f.py
# Topologically Sorted Source Nodes: [loss, add, layer_norm, full_2], Original ATen: [aten.nll_loss_forward, aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   add => add
#   full_2 => full_default_8
#   layer_norm => mul, sub
#   loss => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_532 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_294, torch.float32), kwargs = {})
#   %mul_341 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_532, %primals_4), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, 768), kwargs = {})
#   %sum_77 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_341, [2], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %mul), kwargs = {})
#   %sum_78 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_343, [2], True), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_78), kwargs = {})
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_342, %sum_77), kwargs = {})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_101, %mul_344), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 768), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_26, %sub_102), kwargs = {})
#   %add_110 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %mul_345), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_3, %full_default_1, %add_110), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([65, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_8, [%primals_1], %where_5, True), kwargs = {})
triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_9 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel):
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
    x2 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 768*x2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = 0.0013020833333333333
    tmp22 = tmp13 * tmp21
    tmp23 = 768.0
    tmp24 = tmp3 * tmp23
    tmp25 = tmp24 - tmp7
    tmp26 = tmp14 * tmp19
    tmp27 = tmp25 - tmp26
    tmp28 = tmp22 * tmp27
    tmp29 = tmp20 + tmp28
    tmp31 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tl.device_assert((0 <= tmp34) & (tmp34 < 65), "index out of bounds: 0 <= tmp34 < 65")
    tmp36 = tl.full([1], -1, tl.int64)
    tmp37 = tmp30 == tmp36
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp29)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp29, r0_mask)
    tl.atomic_add(out_ptr2 + (tl.broadcast_to(r0_1 + 768*tmp34, [R0_BLOCK])), tmp39, r0_mask, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/u5/cu5iwly376ta7q7pt4c4mqhh4jpoxuyx5kyrws22p5fokyblzmia.py
# Topologically Sorted Source Nodes: [add, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   layer_norm => mul, sub
# Graph fragment:
#   %convert_element_type_532 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_294, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_532, %mul), kwargs = {})
#   %sum_79 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_346, [0, 1]), kwargs = {})
triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 98304*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*r0_2 + 98304*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + 768*((r0_2 % 64))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr4 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 - tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/pq/cpqry5jt5gqo2nwo7llcdfjd5iblguftzidsffunmf6fp5hwvytw.py
# Topologically Sorted Source Nodes: [loss, full_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_1 => full_default_6
#   loss => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_2, %full_default_1, %view_295), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_6, [%iota], %where_4, True), kwargs = {})
triton_poi_fused_embedding_dense_backward_nll_loss_forward_11 = async_compile.triton('triton_poi_fused_embedding_dense_backward_nll_loss_forward_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_11', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_nll_loss_forward_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 768
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp9 = tl.load(in_ptr1 + (49152 + x2), None)
    tmp11 = tl.load(in_ptr1 + (98304 + x2), None)
    tmp13 = tl.load(in_ptr1 + (147456 + x2), None)
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 64), "index out of bounds: 0 <= tmp4 < 64")
    tmp6 = tl.full([1], -1, tl.int64)
    tmp7 = tmp0 == tmp6
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp7, tmp15, tmp14)
    tl.atomic_add(out_ptr0 + (x0 + 768*tmp4), tmp16, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/3k/c3kl5bmexxwgdw73uxmqczlxrlwkgqjrichsdybgqbsaki6bgped.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_209 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_49, torch.float32), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_209, %index_put_1), kwargs = {})
triton_poi_fused__to_copy_add_12 = async_compile.triton('triton_poi_fused__to_copy_add_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/vm/cvmaf2xmjysnulxalbhkmqohd3hsr6be2myj3olggfo7mfaqp3dc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_230 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_55, torch.float32), kwargs = {})
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/vf/cvfsumescgatghnaq6szcolwuqtn5eol3c2zuugmlme6vrcuiccu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_236 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_57, torch.float32), kwargs = {})
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/r6/cr6xmxhq4pezl6hzgzdyxhf5pktwi3ufqhljkwzzkuqc2zyo2mgu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_215 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_51, torch.float32), kwargs = {})
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_4, primals_7, primals_10, primals_13, primals_16, primals_19, primals_22, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_77, iota, embedding, embedding_1, getitem_1, rsqrt, view, permute_1, permute_2, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mul_2, view_8, mm_2, view_10, mul_7, view_12, permute_9, permute_10, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_9, view_20, mm_6, view_22, mul_14, view_24, permute_17, permute_18, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_16, view_32, mm_10, view_34, mul_21, view_36, permute_25, permute_26, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_23, view_44, mm_14, view_46, mul_28, view_48, permute_33, permute_34, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_30, view_56, mm_18, view_58, mul_35, view_60, permute_41, permute_42, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_37, view_68, mm_22, view_70, mul_42, view_72, permute_49, permute_50, permute_51, getitem_101, getitem_102, getitem_107, getitem_108, mul_44, view_80, mm_26, view_82, mul_49, view_84, permute_57, permute_58, permute_59, getitem_117, getitem_118, getitem_123, getitem_124, mul_51, view_92, mm_30, view_94, mul_56, view_96, permute_65, permute_66, permute_67, getitem_133, getitem_134, getitem_139, getitem_140, mul_58, view_104, mm_34, view_106, mul_63, view_108, permute_73, permute_74, permute_75, getitem_149, getitem_150, getitem_155, getitem_156, mul_65, view_116, mm_38, view_118, mul_70, view_120, permute_81, permute_82, permute_83, getitem_165, getitem_166, getitem_171, getitem_172, mul_72, view_128, mm_42, view_130, mul_77, view_132, permute_89, permute_90, permute_91, getitem_181, getitem_182, getitem_187, getitem_188, mul_79, view_140, mm_46, view_142, mul_84, view_144, view_145, amax, log, convert_element_type_199, permute_99, div_2, permute_103, permute_107, div_3, permute_111, permute_119, div_4, permute_123, permute_127, div_5, permute_131, permute_139, div_6, permute_143, permute_147, div_7, permute_151, permute_159, div_8, permute_163, permute_167, div_9, permute_171, permute_179, div_10, permute_183, permute_187, div_11, permute_191, permute_199, div_12, permute_203, permute_207, div_13, permute_211, permute_219, div_14, permute_223, permute_227, div_15, permute_231, permute_239, div_16, permute_243, permute_247, div_17, permute_251, permute_259, div_18, permute_263, permute_267, div_19, permute_271, permute_279, div_20, permute_283, permute_287, div_21, permute_291, permute_299, div_22, permute_303, permute_307, div_23, permute_311, permute_319, div_24, permute_323, permute_327, div_25, permute_331, permute_339, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 64), (64, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (4, 64), (64, 1))
    assert_size_stride(iota, (64, ), (1, ))
    assert_size_stride(embedding, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(embedding_1, (64, 768), (768, 1))
    assert_size_stride(getitem_1, (4, 64, 1), (64, 1, 1))
    assert_size_stride(rsqrt, (4, 64, 1), (64, 1, 1))
    assert_size_stride(view, (256, 768), (768, 1))
    assert_size_stride(permute_1, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_2, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_3, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_5, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_6, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_11, (2, ), (1, ))
    assert_size_stride(getitem_12, (), ())
    assert_size_stride(mul_2, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_8, (256, 768), (768, 1))
    assert_size_stride(mm_2, (256, 3072), (3072, 1))
    assert_size_stride(view_10, (256, 3072), (3072, 1))
    assert_size_stride(mul_7, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_12, (256, 768), (768, 1))
    assert_size_stride(permute_9, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_10, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_11, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_21, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_22, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_27, (2, ), (1, ))
    assert_size_stride(getitem_28, (), ())
    assert_size_stride(mul_9, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_20, (256, 768), (768, 1))
    assert_size_stride(mm_6, (256, 3072), (3072, 1))
    assert_size_stride(view_22, (256, 3072), (3072, 1))
    assert_size_stride(mul_14, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_24, (256, 768), (768, 1))
    assert_size_stride(permute_17, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_18, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_19, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_37, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_38, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_43, (2, ), (1, ))
    assert_size_stride(getitem_44, (), ())
    assert_size_stride(mul_16, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_32, (256, 768), (768, 1))
    assert_size_stride(mm_10, (256, 3072), (3072, 1))
    assert_size_stride(view_34, (256, 3072), (3072, 1))
    assert_size_stride(mul_21, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_36, (256, 768), (768, 1))
    assert_size_stride(permute_25, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_26, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_27, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_53, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_54, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_59, (2, ), (1, ))
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(mul_23, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_44, (256, 768), (768, 1))
    assert_size_stride(mm_14, (256, 3072), (3072, 1))
    assert_size_stride(view_46, (256, 3072), (3072, 1))
    assert_size_stride(mul_28, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_48, (256, 768), (768, 1))
    assert_size_stride(permute_33, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_34, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_35, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_69, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_70, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_75, (2, ), (1, ))
    assert_size_stride(getitem_76, (), ())
    assert_size_stride(mul_30, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_56, (256, 768), (768, 1))
    assert_size_stride(mm_18, (256, 3072), (3072, 1))
    assert_size_stride(view_58, (256, 3072), (3072, 1))
    assert_size_stride(mul_35, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_60, (256, 768), (768, 1))
    assert_size_stride(permute_41, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_42, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_43, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_85, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_86, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_91, (2, ), (1, ))
    assert_size_stride(getitem_92, (), ())
    assert_size_stride(mul_37, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_68, (256, 768), (768, 1))
    assert_size_stride(mm_22, (256, 3072), (3072, 1))
    assert_size_stride(view_70, (256, 3072), (3072, 1))
    assert_size_stride(mul_42, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_72, (256, 768), (768, 1))
    assert_size_stride(permute_49, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_50, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_51, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_101, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_102, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_107, (2, ), (1, ))
    assert_size_stride(getitem_108, (), ())
    assert_size_stride(mul_44, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_80, (256, 768), (768, 1))
    assert_size_stride(mm_26, (256, 3072), (3072, 1))
    assert_size_stride(view_82, (256, 3072), (3072, 1))
    assert_size_stride(mul_49, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_84, (256, 768), (768, 1))
    assert_size_stride(permute_57, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_58, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_59, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_117, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_118, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_123, (2, ), (1, ))
    assert_size_stride(getitem_124, (), ())
    assert_size_stride(mul_51, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_92, (256, 768), (768, 1))
    assert_size_stride(mm_30, (256, 3072), (3072, 1))
    assert_size_stride(view_94, (256, 3072), (3072, 1))
    assert_size_stride(mul_56, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_96, (256, 768), (768, 1))
    assert_size_stride(permute_65, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_66, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_67, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_133, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_134, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_139, (2, ), (1, ))
    assert_size_stride(getitem_140, (), ())
    assert_size_stride(mul_58, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_104, (256, 768), (768, 1))
    assert_size_stride(mm_34, (256, 3072), (3072, 1))
    assert_size_stride(view_106, (256, 3072), (3072, 1))
    assert_size_stride(mul_63, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_108, (256, 768), (768, 1))
    assert_size_stride(permute_73, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_74, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_75, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_149, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_150, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_155, (2, ), (1, ))
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(mul_65, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_116, (256, 768), (768, 1))
    assert_size_stride(mm_38, (256, 3072), (3072, 1))
    assert_size_stride(view_118, (256, 3072), (3072, 1))
    assert_size_stride(mul_70, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_120, (256, 768), (768, 1))
    assert_size_stride(permute_81, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_82, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_83, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_165, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_166, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_171, (2, ), (1, ))
    assert_size_stride(getitem_172, (), ())
    assert_size_stride(mul_72, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_128, (256, 768), (768, 1))
    assert_size_stride(mm_42, (256, 3072), (3072, 1))
    assert_size_stride(view_130, (256, 3072), (3072, 1))
    assert_size_stride(mul_77, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_132, (256, 768), (768, 1))
    assert_size_stride(permute_89, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_90, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(permute_91, (4, 12, 64, 64), (147456, 64, 2304, 1))
    assert_size_stride(getitem_181, (4, 12, 64, 64), (49152, 64, 768, 1))
    assert_size_stride(getitem_182, (4, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_187, (2, ), (1, ))
    assert_size_stride(getitem_188, (), ())
    assert_size_stride(mul_79, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_140, (256, 768), (768, 1))
    assert_size_stride(mm_46, (256, 3072), (3072, 1))
    assert_size_stride(view_142, (256, 3072), (3072, 1))
    assert_size_stride(mul_84, (4, 64, 768), (49152, 768, 1))
    assert_size_stride(view_144, (256, 768), (768, 1))
    assert_size_stride(view_145, (4, 64, 65), (4160, 65, 1))
    assert_size_stride(amax, (256, 1), (1, 1))
    assert_size_stride(log, (256, 1), (1, 1))
    assert_size_stride(convert_element_type_199, (), ())
    assert_size_stride(permute_99, (65, 768), (768, 1))
    assert_size_stride(div_2, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_103, (768, 3072), (3072, 1))
    assert_size_stride(permute_107, (3072, 768), (768, 1))
    assert_size_stride(div_3, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_111, (768, 768), (768, 1))
    assert_size_stride(permute_119, (2304, 768), (768, 1))
    assert_size_stride(div_4, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_123, (768, 3072), (3072, 1))
    assert_size_stride(permute_127, (3072, 768), (768, 1))
    assert_size_stride(div_5, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_131, (768, 768), (768, 1))
    assert_size_stride(permute_139, (2304, 768), (768, 1))
    assert_size_stride(div_6, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_143, (768, 3072), (3072, 1))
    assert_size_stride(permute_147, (3072, 768), (768, 1))
    assert_size_stride(div_7, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_159, (2304, 768), (768, 1))
    assert_size_stride(div_8, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_163, (768, 3072), (3072, 1))
    assert_size_stride(permute_167, (3072, 768), (768, 1))
    assert_size_stride(div_9, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(permute_179, (2304, 768), (768, 1))
    assert_size_stride(div_10, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_183, (768, 3072), (3072, 1))
    assert_size_stride(permute_187, (3072, 768), (768, 1))
    assert_size_stride(div_11, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_191, (768, 768), (768, 1))
    assert_size_stride(permute_199, (2304, 768), (768, 1))
    assert_size_stride(div_12, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_203, (768, 3072), (3072, 1))
    assert_size_stride(permute_207, (3072, 768), (768, 1))
    assert_size_stride(div_13, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_211, (768, 768), (768, 1))
    assert_size_stride(permute_219, (2304, 768), (768, 1))
    assert_size_stride(div_14, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_223, (768, 3072), (3072, 1))
    assert_size_stride(permute_227, (3072, 768), (768, 1))
    assert_size_stride(div_15, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_231, (768, 768), (768, 1))
    assert_size_stride(permute_239, (2304, 768), (768, 1))
    assert_size_stride(div_16, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_243, (768, 3072), (3072, 1))
    assert_size_stride(permute_247, (3072, 768), (768, 1))
    assert_size_stride(div_17, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_251, (768, 768), (768, 1))
    assert_size_stride(permute_259, (2304, 768), (768, 1))
    assert_size_stride(div_18, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_263, (768, 3072), (3072, 1))
    assert_size_stride(permute_267, (3072, 768), (768, 1))
    assert_size_stride(div_19, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_271, (768, 768), (768, 1))
    assert_size_stride(permute_279, (2304, 768), (768, 1))
    assert_size_stride(div_20, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_283, (768, 3072), (3072, 1))
    assert_size_stride(permute_287, (3072, 768), (768, 1))
    assert_size_stride(div_21, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_291, (768, 768), (768, 1))
    assert_size_stride(permute_299, (2304, 768), (768, 1))
    assert_size_stride(div_22, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_303, (768, 3072), (3072, 1))
    assert_size_stride(permute_307, (3072, 768), (768, 1))
    assert_size_stride(div_23, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_311, (768, 768), (768, 1))
    assert_size_stride(permute_319, (2304, 768), (768, 1))
    assert_size_stride(div_24, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_323, (768, 3072), (3072, 1))
    assert_size_stride(permute_327, (3072, 768), (768, 1))
    assert_size_stride(div_25, (4, 64, 1), (64, 1, 1))
    assert_size_stride(permute_331, (768, 768), (768, 1))
    assert_size_stride(permute_339, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (4, 64, 65), (4160, 65, 1))
    assert_size_stride(tangents_2, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 64, 65), (4160, 65, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [loss, scatter, convert_element_type_default], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_0.run(primals_77, tangents_2, convert_element_type_199, tangents_1, view_145, amax, log, buf1, 256, 65, stream=stream0)
        del amax
        del convert_element_type_199
        del log
        del primals_77
        del tangents_1
        del tangents_2
        del view_145
        buf2 = empty_strided_cuda((65, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_100], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (65, 256), (1, 65), 0), view_144, out=buf2)
        del view_144
        buf369 = empty_strided_cuda((64, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_1], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_1.run(buf369, 49152, stream=stream0)
        buf371 = empty_strided_cuda((65, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_2], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_2.run(buf371, 49920, stream=stream0)
        buf3 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (256, 65), (65, 1), 0), permute_99, out=buf3)
        del buf1
        del permute_99
        buf7 = empty_strided_cuda((768, 2), (1, 768), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf3, mul_84, buf7, 1536, 128, stream=stream0)
        buf8 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf7, buf8, 768, 2, stream=stream0)
        buf6 = mul_84; del mul_84  # reuse
        buf9 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_view_5.run(buf6, buf3, primals_76, div_2, buf9, 256, 768, stream=stream0)
        del div_2
        del primals_76
        buf11 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, permute_103, out=buf11)
        del permute_103
        buf13 = reinterpret_tensor(buf11, (4, 64, 3072), (196608, 3072, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, mm_46, 786432, stream=stream0)
        del mm_46
        buf15 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (256, 3072), (3072, 1), 0), permute_107, out=buf15)
        del permute_107
        buf19 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf15, mul_79, buf19, 1536, 128, stream=stream0)
        buf21 = buf6; del buf6  # reuse
        buf22 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf21, buf15, primals_73, mul_79, div_3, buf22, 256, 768, stream=stream0)
        del div_3
        del mul_79
        del primals_73
        buf24 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, permute_111, out=buf24)
        del permute_111
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf26 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf24, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_90, permute_89, permute_91, getitem_181, getitem_182, None, None, 64, 64, 0.0, True, getitem_187, getitem_188, scale=0.125)
        del buf24
        del getitem_182
        del getitem_187
        del getitem_188
        del permute_89
        del permute_90
        del permute_91
        buf20 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf19, buf20, 768, 2, stream=stream0)
        buf27 = buf26[0]
        assert_size_stride(buf27, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf28 = buf26[1]
        assert_size_stride(buf28, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf29 = buf26[2]
        assert_size_stride(buf29, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf26
        buf30 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf27, buf28, buf29, buf30, 589824, stream=stream0)
        buf32 = reinterpret_tensor(buf29, (256, 768), (768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (256, 2304), (2304, 1), 0), permute_119, out=buf32)
        del permute_119
        buf36 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf32, mul_77, buf36, 1536, 128, stream=stream0)
        buf38 = buf21; del buf21  # reuse
        buf39 = reinterpret_tensor(buf28, (256, 768), (768, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf38, buf32, primals_70, mul_77, div_4, buf39, 256, 768, stream=stream0)
        del div_4
        del mul_77
        del primals_70
        buf41 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, permute_123, out=buf41)
        del permute_123
        buf43 = reinterpret_tensor(buf41, (4, 64, 3072), (196608, 3072, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf43, mm_42, 786432, stream=stream0)
        del mm_42
        buf45 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (256, 3072), (3072, 1), 0), permute_127, out=buf45)
        del permute_127
        buf37 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf36, buf37, 768, 2, stream=stream0)
        buf49 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf45, mul_72, buf49, 1536, 128, stream=stream0)
        buf51 = buf38; del buf38  # reuse
        buf52 = reinterpret_tensor(buf27, (256, 768), (768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf51, buf45, primals_67, mul_72, div_5, buf52, 256, 768, stream=stream0)
        del div_5
        del mul_72
        del primals_67
        buf54 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, permute_131, out=buf54)
        del permute_131
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf56 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf54, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_82, permute_81, permute_83, getitem_165, getitem_166, None, None, 64, 64, 0.0, True, getitem_171, getitem_172, scale=0.125)
        del buf54
        del getitem_166
        del getitem_171
        del getitem_172
        del permute_81
        del permute_82
        del permute_83
        buf50 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf49, buf50, 768, 2, stream=stream0)
        buf57 = buf56[0]
        assert_size_stride(buf57, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf58 = buf56[1]
        assert_size_stride(buf58, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf59 = buf56[2]
        assert_size_stride(buf59, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf56
        buf60 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf57, buf58, buf59, buf60, 589824, stream=stream0)
        buf62 = reinterpret_tensor(buf59, (256, 768), (768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (256, 2304), (2304, 1), 0), permute_139, out=buf62)
        del permute_139
        buf66 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf62, mul_70, buf66, 1536, 128, stream=stream0)
        buf68 = buf51; del buf51  # reuse
        buf69 = reinterpret_tensor(buf58, (256, 768), (768, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf68, buf62, primals_64, mul_70, div_6, buf69, 256, 768, stream=stream0)
        del div_6
        del mul_70
        del primals_64
        buf71 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf69, permute_143, out=buf71)
        del permute_143
        buf73 = reinterpret_tensor(buf71, (4, 64, 3072), (196608, 3072, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf73, mm_38, 786432, stream=stream0)
        del mm_38
        buf75 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (256, 3072), (3072, 1), 0), permute_147, out=buf75)
        del permute_147
        buf67 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf66, buf67, 768, 2, stream=stream0)
        buf79 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf75, mul_65, buf79, 1536, 128, stream=stream0)
        buf81 = buf68; del buf68  # reuse
        buf82 = reinterpret_tensor(buf57, (256, 768), (768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf81, buf75, primals_61, mul_65, div_7, buf82, 256, 768, stream=stream0)
        del div_7
        del mul_65
        del primals_61
        buf84 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf82, permute_151, out=buf84)
        del permute_151
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf86 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf84, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_74, permute_73, permute_75, getitem_149, getitem_150, None, None, 64, 64, 0.0, True, getitem_155, getitem_156, scale=0.125)
        del buf84
        del getitem_150
        del getitem_155
        del getitem_156
        del permute_73
        del permute_74
        del permute_75
        buf80 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf79, buf80, 768, 2, stream=stream0)
        buf87 = buf86[0]
        assert_size_stride(buf87, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf88 = buf86[1]
        assert_size_stride(buf88, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf89 = buf86[2]
        assert_size_stride(buf89, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf86
        buf90 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf87, buf88, buf89, buf90, 589824, stream=stream0)
        buf92 = reinterpret_tensor(buf89, (256, 768), (768, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (256, 2304), (2304, 1), 0), permute_159, out=buf92)
        del permute_159
        buf96 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf92, mul_63, buf96, 1536, 128, stream=stream0)
        buf98 = buf81; del buf81  # reuse
        buf99 = reinterpret_tensor(buf88, (256, 768), (768, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf98, buf92, primals_58, mul_63, div_8, buf99, 256, 768, stream=stream0)
        del div_8
        del mul_63
        del primals_58
        buf101 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf99, permute_163, out=buf101)
        del permute_163
        buf103 = reinterpret_tensor(buf101, (4, 64, 3072), (196608, 3072, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf103, mm_34, 786432, stream=stream0)
        del mm_34
        buf105 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (256, 3072), (3072, 1), 0), permute_167, out=buf105)
        del permute_167
        buf97 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf96, buf97, 768, 2, stream=stream0)
        buf109 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf105, mul_58, buf109, 1536, 128, stream=stream0)
        buf111 = buf98; del buf98  # reuse
        buf112 = reinterpret_tensor(buf87, (256, 768), (768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf111, buf105, primals_55, mul_58, div_9, buf112, 256, 768, stream=stream0)
        del div_9
        del mul_58
        del primals_55
        buf114 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, permute_171, out=buf114)
        del permute_171
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf116 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf114, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_66, permute_65, permute_67, getitem_133, getitem_134, None, None, 64, 64, 0.0, True, getitem_139, getitem_140, scale=0.125)
        del buf114
        del getitem_134
        del getitem_139
        del getitem_140
        del permute_65
        del permute_66
        del permute_67
        buf110 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf109, buf110, 768, 2, stream=stream0)
        buf117 = buf116[0]
        assert_size_stride(buf117, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf118 = buf116[1]
        assert_size_stride(buf118, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf119 = buf116[2]
        assert_size_stride(buf119, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf116
        buf120 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf117, buf118, buf119, buf120, 589824, stream=stream0)
        buf122 = reinterpret_tensor(buf119, (256, 768), (768, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (256, 2304), (2304, 1), 0), permute_179, out=buf122)
        del permute_179
        buf126 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf122, mul_56, buf126, 1536, 128, stream=stream0)
        buf128 = buf111; del buf111  # reuse
        buf129 = reinterpret_tensor(buf118, (256, 768), (768, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf128, buf122, primals_52, mul_56, div_10, buf129, 256, 768, stream=stream0)
        del div_10
        del mul_56
        del primals_52
        buf131 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf129, permute_183, out=buf131)
        del permute_183
        buf133 = reinterpret_tensor(buf131, (4, 64, 3072), (196608, 3072, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf133, mm_30, 786432, stream=stream0)
        del mm_30
        buf135 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (256, 3072), (3072, 1), 0), permute_187, out=buf135)
        del permute_187
        buf127 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf126, buf127, 768, 2, stream=stream0)
        buf139 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf135, mul_51, buf139, 1536, 128, stream=stream0)
        buf141 = buf128; del buf128  # reuse
        buf142 = reinterpret_tensor(buf117, (256, 768), (768, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf141, buf135, primals_49, mul_51, div_11, buf142, 256, 768, stream=stream0)
        del div_11
        del mul_51
        del primals_49
        buf144 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf142, permute_191, out=buf144)
        del permute_191
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf146 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf144, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_58, permute_57, permute_59, getitem_117, getitem_118, None, None, 64, 64, 0.0, True, getitem_123, getitem_124, scale=0.125)
        del buf144
        del getitem_118
        del getitem_123
        del getitem_124
        del permute_57
        del permute_58
        del permute_59
        buf140 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf139, buf140, 768, 2, stream=stream0)
        buf147 = buf146[0]
        assert_size_stride(buf147, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf148 = buf146[1]
        assert_size_stride(buf148, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf149 = buf146[2]
        assert_size_stride(buf149, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf146
        buf150 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf147, buf148, buf149, buf150, 589824, stream=stream0)
        buf152 = reinterpret_tensor(buf149, (256, 768), (768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (256, 2304), (2304, 1), 0), permute_199, out=buf152)
        del permute_199
        buf156 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf152, mul_49, buf156, 1536, 128, stream=stream0)
        buf158 = buf141; del buf141  # reuse
        buf159 = reinterpret_tensor(buf148, (256, 768), (768, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf158, buf152, primals_46, mul_49, div_12, buf159, 256, 768, stream=stream0)
        del div_12
        del mul_49
        del primals_46
        buf161 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf159, permute_203, out=buf161)
        del permute_203
        buf163 = reinterpret_tensor(buf161, (4, 64, 3072), (196608, 3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf163, mm_26, 786432, stream=stream0)
        del mm_26
        buf165 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (256, 3072), (3072, 1), 0), permute_207, out=buf165)
        del permute_207
        buf157 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf156, buf157, 768, 2, stream=stream0)
        buf169 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf165, mul_44, buf169, 1536, 128, stream=stream0)
        buf171 = buf158; del buf158  # reuse
        buf172 = reinterpret_tensor(buf147, (256, 768), (768, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf171, buf165, primals_43, mul_44, div_13, buf172, 256, 768, stream=stream0)
        del div_13
        del mul_44
        del primals_43
        buf174 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, permute_211, out=buf174)
        del permute_211
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf176 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf174, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_50, permute_49, permute_51, getitem_101, getitem_102, None, None, 64, 64, 0.0, True, getitem_107, getitem_108, scale=0.125)
        del buf174
        del getitem_102
        del getitem_107
        del getitem_108
        del permute_49
        del permute_50
        del permute_51
        buf170 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf169, buf170, 768, 2, stream=stream0)
        buf177 = buf176[0]
        assert_size_stride(buf177, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf178 = buf176[1]
        assert_size_stride(buf178, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf179 = buf176[2]
        assert_size_stride(buf179, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf176
        buf180 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf177, buf178, buf179, buf180, 589824, stream=stream0)
        buf182 = reinterpret_tensor(buf179, (256, 768), (768, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (256, 2304), (2304, 1), 0), permute_219, out=buf182)
        del permute_219
        buf186 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf182, mul_42, buf186, 1536, 128, stream=stream0)
        buf188 = buf171; del buf171  # reuse
        buf189 = reinterpret_tensor(buf178, (256, 768), (768, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf188, buf182, primals_40, mul_42, div_14, buf189, 256, 768, stream=stream0)
        del div_14
        del mul_42
        del primals_40
        buf191 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, permute_223, out=buf191)
        del permute_223
        buf193 = reinterpret_tensor(buf191, (4, 64, 3072), (196608, 3072, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf193, mm_22, 786432, stream=stream0)
        del mm_22
        buf195 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (256, 3072), (3072, 1), 0), permute_227, out=buf195)
        del permute_227
        buf187 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf186, buf187, 768, 2, stream=stream0)
        buf199 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf195, mul_37, buf199, 1536, 128, stream=stream0)
        buf201 = buf188; del buf188  # reuse
        buf202 = reinterpret_tensor(buf177, (256, 768), (768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf201, buf195, primals_37, mul_37, div_15, buf202, 256, 768, stream=stream0)
        del div_15
        del mul_37
        del primals_37
        buf204 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, permute_231, out=buf204)
        del permute_231
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf206 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf204, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_42, permute_41, permute_43, getitem_85, getitem_86, None, None, 64, 64, 0.0, True, getitem_91, getitem_92, scale=0.125)
        del buf204
        del getitem_86
        del getitem_91
        del getitem_92
        del permute_41
        del permute_42
        del permute_43
        buf200 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf199, buf200, 768, 2, stream=stream0)
        buf207 = buf206[0]
        assert_size_stride(buf207, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf208 = buf206[1]
        assert_size_stride(buf208, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf209 = buf206[2]
        assert_size_stride(buf209, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf206
        buf210 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf207, buf208, buf209, buf210, 589824, stream=stream0)
        buf212 = reinterpret_tensor(buf209, (256, 768), (768, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (256, 2304), (2304, 1), 0), permute_239, out=buf212)
        del permute_239
        buf216 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf212, mul_35, buf216, 1536, 128, stream=stream0)
        buf218 = buf201; del buf201  # reuse
        buf219 = reinterpret_tensor(buf208, (256, 768), (768, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf218, buf212, primals_34, mul_35, div_16, buf219, 256, 768, stream=stream0)
        del div_16
        del mul_35
        del primals_34
        buf221 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf219, permute_243, out=buf221)
        del permute_243
        buf223 = reinterpret_tensor(buf221, (4, 64, 3072), (196608, 3072, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf223, mm_18, 786432, stream=stream0)
        del mm_18
        buf225 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (256, 3072), (3072, 1), 0), permute_247, out=buf225)
        del permute_247
        buf217 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf216, buf217, 768, 2, stream=stream0)
        buf229 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf225, mul_30, buf229, 1536, 128, stream=stream0)
        buf231 = buf218; del buf218  # reuse
        buf232 = reinterpret_tensor(buf207, (256, 768), (768, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf231, buf225, primals_31, mul_30, div_17, buf232, 256, 768, stream=stream0)
        del div_17
        del mul_30
        del primals_31
        buf234 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf232, permute_251, out=buf234)
        del permute_251
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf236 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf234, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_34, permute_33, permute_35, getitem_69, getitem_70, None, None, 64, 64, 0.0, True, getitem_75, getitem_76, scale=0.125)
        del buf234
        del getitem_70
        del getitem_75
        del getitem_76
        del permute_33
        del permute_34
        del permute_35
        buf230 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf229, buf230, 768, 2, stream=stream0)
        buf237 = buf236[0]
        assert_size_stride(buf237, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf238 = buf236[1]
        assert_size_stride(buf238, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf239 = buf236[2]
        assert_size_stride(buf239, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf236
        buf240 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf237, buf238, buf239, buf240, 589824, stream=stream0)
        buf242 = reinterpret_tensor(buf239, (256, 768), (768, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (256, 2304), (2304, 1), 0), permute_259, out=buf242)
        del permute_259
        buf246 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf242, mul_28, buf246, 1536, 128, stream=stream0)
        buf248 = buf231; del buf231  # reuse
        buf249 = reinterpret_tensor(buf238, (256, 768), (768, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf248, buf242, primals_28, mul_28, div_18, buf249, 256, 768, stream=stream0)
        del div_18
        del mul_28
        del primals_28
        buf251 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf249, permute_263, out=buf251)
        del permute_263
        buf253 = reinterpret_tensor(buf251, (4, 64, 3072), (196608, 3072, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf253, mm_14, 786432, stream=stream0)
        del mm_14
        buf255 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (256, 3072), (3072, 1), 0), permute_267, out=buf255)
        del permute_267
        buf247 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf246, buf247, 768, 2, stream=stream0)
        buf259 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf255, mul_23, buf259, 1536, 128, stream=stream0)
        buf261 = buf248; del buf248  # reuse
        buf262 = reinterpret_tensor(buf237, (256, 768), (768, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf261, buf255, primals_25, mul_23, div_19, buf262, 256, 768, stream=stream0)
        del div_19
        del mul_23
        del primals_25
        buf264 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf262, permute_271, out=buf264)
        del permute_271
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf266 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf264, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_26, permute_25, permute_27, getitem_53, getitem_54, None, None, 64, 64, 0.0, True, getitem_59, getitem_60, scale=0.125)
        del buf264
        del getitem_54
        del getitem_59
        del getitem_60
        del permute_25
        del permute_26
        del permute_27
        buf260 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf259, buf260, 768, 2, stream=stream0)
        buf267 = buf266[0]
        assert_size_stride(buf267, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf268 = buf266[1]
        assert_size_stride(buf268, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf269 = buf266[2]
        assert_size_stride(buf269, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf266
        buf270 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf267, buf268, buf269, buf270, 589824, stream=stream0)
        buf272 = reinterpret_tensor(buf269, (256, 768), (768, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (256, 2304), (2304, 1), 0), permute_279, out=buf272)
        del permute_279
        buf276 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf272, mul_21, buf276, 1536, 128, stream=stream0)
        buf278 = buf261; del buf261  # reuse
        buf279 = reinterpret_tensor(buf268, (256, 768), (768, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf278, buf272, primals_22, mul_21, div_20, buf279, 256, 768, stream=stream0)
        del div_20
        del mul_21
        del primals_22
        buf281 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf279, permute_283, out=buf281)
        del permute_283
        buf283 = reinterpret_tensor(buf281, (4, 64, 3072), (196608, 3072, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf283, mm_10, 786432, stream=stream0)
        del mm_10
        buf285 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (256, 3072), (3072, 1), 0), permute_287, out=buf285)
        del permute_287
        buf277 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf276, buf277, 768, 2, stream=stream0)
        buf289 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf285, mul_16, buf289, 1536, 128, stream=stream0)
        buf291 = buf278; del buf278  # reuse
        buf292 = reinterpret_tensor(buf267, (256, 768), (768, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf291, buf285, primals_19, mul_16, div_21, buf292, 256, 768, stream=stream0)
        del div_21
        del mul_16
        del primals_19
        buf294 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, permute_291, out=buf294)
        del permute_291
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf296 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf294, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_18, permute_17, permute_19, getitem_37, getitem_38, None, None, 64, 64, 0.0, True, getitem_43, getitem_44, scale=0.125)
        del buf294
        del getitem_38
        del getitem_43
        del getitem_44
        del permute_17
        del permute_18
        del permute_19
        buf290 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf289, buf290, 768, 2, stream=stream0)
        buf297 = buf296[0]
        assert_size_stride(buf297, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf298 = buf296[1]
        assert_size_stride(buf298, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf299 = buf296[2]
        assert_size_stride(buf299, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf296
        buf300 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf297, buf298, buf299, buf300, 589824, stream=stream0)
        buf302 = reinterpret_tensor(buf299, (256, 768), (768, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (256, 2304), (2304, 1), 0), permute_299, out=buf302)
        del permute_299
        buf306 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf302, mul_14, buf306, 1536, 128, stream=stream0)
        buf308 = buf291; del buf291  # reuse
        buf309 = reinterpret_tensor(buf298, (256, 768), (768, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf308, buf302, primals_16, mul_14, div_22, buf309, 256, 768, stream=stream0)
        del div_22
        del mul_14
        del primals_16
        buf311 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, permute_303, out=buf311)
        del permute_303
        buf313 = reinterpret_tensor(buf311, (4, 64, 3072), (196608, 3072, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf313, mm_6, 786432, stream=stream0)
        del mm_6
        buf315 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (256, 3072), (3072, 1), 0), permute_307, out=buf315)
        del permute_307
        buf307 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf306, buf307, 768, 2, stream=stream0)
        buf319 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf315, mul_9, buf319, 1536, 128, stream=stream0)
        buf321 = buf308; del buf308  # reuse
        buf322 = reinterpret_tensor(buf297, (256, 768), (768, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf321, buf315, primals_13, mul_9, div_23, buf322, 256, 768, stream=stream0)
        del div_23
        del mul_9
        del primals_13
        buf324 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf322, permute_311, out=buf324)
        del permute_311
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf326 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf324, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_10, permute_9, permute_11, getitem_21, getitem_22, None, None, 64, 64, 0.0, True, getitem_27, getitem_28, scale=0.125)
        del buf324
        del getitem_22
        del getitem_27
        del getitem_28
        del permute_10
        del permute_11
        del permute_9
        buf320 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf319, buf320, 768, 2, stream=stream0)
        buf327 = buf326[0]
        assert_size_stride(buf327, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf328 = buf326[1]
        assert_size_stride(buf328, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf329 = buf326[2]
        assert_size_stride(buf329, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf326
        buf330 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf327, buf328, buf329, buf330, 589824, stream=stream0)
        buf332 = reinterpret_tensor(buf329, (256, 768), (768, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (256, 2304), (2304, 1), 0), permute_319, out=buf332)
        del permute_319
        buf336 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf332, mul_7, buf336, 1536, 128, stream=stream0)
        buf338 = buf321; del buf321  # reuse
        buf339 = reinterpret_tensor(buf328, (256, 768), (768, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf338, buf332, primals_10, mul_7, div_24, buf339, 256, 768, stream=stream0)
        del div_24
        del mul_7
        del primals_10
        buf341 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf339, permute_323, out=buf341)
        del permute_323
        buf343 = reinterpret_tensor(buf341, (4, 64, 3072), (196608, 3072, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_6.run(buf343, mm_2, 786432, stream=stream0)
        del mm_2
        buf345 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (256, 3072), (3072, 1), 0), permute_327, out=buf345)
        del permute_327
        buf337 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf336, buf337, 768, 2, stream=stream0)
        buf349 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_3.run(buf345, mul_2, buf349, 1536, 128, stream=stream0)
        buf351 = buf338; del buf338  # reuse
        buf352 = reinterpret_tensor(buf327, (256, 768), (768, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_7.run(buf351, buf345, primals_7, mul_2, div_25, buf352, 256, 768, stream=stream0)
        del div_25
        del mul_2
        del primals_7
        buf354 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf352, permute_331, out=buf354)
        del permute_331
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf356 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf354, (4, 12, 64, 64), (49152, 64, 768, 1), 0), permute_2, permute_1, permute_3, getitem_5, getitem_6, None, None, 64, 64, 0.0, True, getitem_11, getitem_12, scale=0.125)
        del buf354
        del getitem_11
        del getitem_12
        del getitem_6
        del permute_1
        del permute_2
        del permute_3
        buf350 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf349, buf350, 768, 2, stream=stream0)
        buf357 = buf356[0]
        assert_size_stride(buf357, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf358 = buf356[1]
        assert_size_stride(buf358, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf359 = buf356[2]
        assert_size_stride(buf359, (4, 12, 64, 64), (49152, 64, 768, 1))
        del buf356
        buf360 = empty_strided_cuda((4, 64, 2304), (147456, 2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf357, buf358, buf359, buf360, 589824, stream=stream0)
        del buf357
        del buf358
        buf362 = reinterpret_tensor(buf359, (256, 768), (768, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (256, 2304), (2304, 1), 0), permute_339, out=buf362)
        del permute_339
        buf368 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [loss, add, layer_norm, full_2], Original ATen: [aten.nll_loss_forward, aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_9.run(buf368, buf362, primals_4, embedding, embedding_1, getitem_1, rsqrt, primals_1, buf371, 256, 768, stream=stream0)
        del primals_1
        del primals_4
        buf366 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [add, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf362, embedding, embedding_1, getitem_1, rsqrt, buf366, 1536, 128, stream=stream0)
        del buf362
        del embedding
        del embedding_1
        del getitem_1
        del rsqrt
        # Topologically Sorted Source Nodes: [loss, full_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_11.run(iota, buf368, buf369, 49152, stream=stream0)
        del buf368
        del iota
        buf373 = empty_strided_cuda((65, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_12.run(buf2, buf371, buf373, 49920, stream=stream0)
        del buf2
        del buf371
        buf367 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [add, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_4.run(buf366, buf367, 768, 2, stream=stream0)
        del buf366
        buf23 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_112], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_181, (256, 768), (768, 1), 0), out=buf23)
        del buf22
        del getitem_181
        buf53 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_132], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_165, (256, 768), (768, 1), 0), out=buf53)
        del buf52
        del getitem_165
        buf83 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_149, (256, 768), (768, 1), 0), out=buf83)
        del buf82
        del getitem_149
        buf113 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_172], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_133, (256, 768), (768, 1), 0), out=buf113)
        del buf112
        del getitem_133
        buf143 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_192], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_117, (256, 768), (768, 1), 0), out=buf143)
        del buf142
        del getitem_117
        buf173 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_212], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_101, (256, 768), (768, 1), 0), out=buf173)
        del buf172
        del getitem_101
        buf203 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_232], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_85, (256, 768), (768, 1), 0), out=buf203)
        del buf202
        del getitem_85
        buf233 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_252], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_69, (256, 768), (768, 1), 0), out=buf233)
        del buf232
        del getitem_69
        buf263 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_272], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_53, (256, 768), (768, 1), 0), out=buf263)
        del buf262
        del getitem_53
        buf293 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_292], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_37, (256, 768), (768, 1), 0), out=buf293)
        del buf292
        del getitem_37
        buf323 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_312], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_21, (256, 768), (768, 1), 0), out=buf323)
        del buf322
        del getitem_21
        buf353 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_332], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (768, 256), (1, 768), 0), reinterpret_tensor(getitem_5, (256, 768), (768, 1), 0), out=buf353)
        del buf352
        del getitem_5
        buf25 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf23, buf25, 589824, stream=stream0)
        del buf23
        buf55 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf53, buf55, 589824, stream=stream0)
        del buf53
        buf85 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf83, buf85, 589824, stream=stream0)
        del buf83
        buf115 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf113, buf115, 589824, stream=stream0)
        del buf113
        buf145 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf143, buf145, 589824, stream=stream0)
        del buf143
        buf175 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf173, buf175, 589824, stream=stream0)
        del buf173
        buf205 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf203, buf205, 589824, stream=stream0)
        del buf203
        buf235 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf233, buf235, 589824, stream=stream0)
        del buf233
        buf265 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf263, buf265, 589824, stream=stream0)
        del buf263
        buf295 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf293, buf295, 589824, stream=stream0)
        del buf293
        buf325 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf323, buf325, 589824, stream=stream0)
        del buf323
        buf355 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf353, buf355, 589824, stream=stream0)
        del buf353
        buf31 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (2304, 256), (1, 2304), 0), view_132, out=buf31)
        del buf30
        del view_132
        buf61 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_140], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (2304, 256), (1, 2304), 0), view_120, out=buf61)
        del buf60
        del view_120
        buf91 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_160], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (2304, 256), (1, 2304), 0), view_108, out=buf91)
        del buf90
        del view_108
        buf121 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_180], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (2304, 256), (1, 2304), 0), view_96, out=buf121)
        del buf120
        del view_96
        buf151 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_200], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (2304, 256), (1, 2304), 0), view_84, out=buf151)
        del buf150
        del view_84
        buf181 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_220], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (2304, 256), (1, 2304), 0), view_72, out=buf181)
        del buf180
        del view_72
        buf211 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_240], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (2304, 256), (1, 2304), 0), view_60, out=buf211)
        del buf210
        del view_60
        buf241 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_260], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (2304, 256), (1, 2304), 0), view_48, out=buf241)
        del buf240
        del view_48
        buf271 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_280], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (2304, 256), (1, 2304), 0), view_36, out=buf271)
        del buf270
        del view_36
        buf301 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_300], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (2304, 256), (1, 2304), 0), view_24, out=buf301)
        del buf300
        del view_24
        buf331 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_320], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (2304, 256), (1, 2304), 0), view_12, out=buf331)
        del buf330
        del view_12
        buf361 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_340], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (2304, 256), (1, 2304), 0), view, out=buf361)
        del buf360
        del view
        buf10 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (768, 256), (1, 768), 0), view_142, out=buf10)
        del buf9
        del view_142
        buf14 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_108], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 256), (1, 3072), 0), view_140, out=buf14)
        del buf13
        del view_140
        buf40 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_124], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (768, 256), (1, 768), 0), view_130, out=buf40)
        del buf39
        del view_130
        buf44 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (3072, 256), (1, 3072), 0), view_128, out=buf44)
        del buf43
        del view_128
        buf70 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (768, 256), (1, 768), 0), view_118, out=buf70)
        del buf69
        del view_118
        buf74 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_148], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (3072, 256), (1, 3072), 0), view_116, out=buf74)
        del buf73
        del view_116
        buf100 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_164], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 256), (1, 768), 0), view_106, out=buf100)
        del buf99
        del view_106
        buf104 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_168], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (3072, 256), (1, 3072), 0), view_104, out=buf104)
        del buf103
        del view_104
        buf130 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_184], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (768, 256), (1, 768), 0), view_94, out=buf130)
        del buf129
        del view_94
        buf134 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_188], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (3072, 256), (1, 3072), 0), view_92, out=buf134)
        del buf133
        del view_92
        buf160 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_204], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (768, 256), (1, 768), 0), view_82, out=buf160)
        del buf159
        del view_82
        buf164 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_208], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (3072, 256), (1, 3072), 0), view_80, out=buf164)
        del buf163
        del view_80
        buf190 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_224], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (768, 256), (1, 768), 0), view_70, out=buf190)
        del buf189
        del view_70
        buf194 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_228], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (3072, 256), (1, 3072), 0), view_68, out=buf194)
        del buf193
        del view_68
        buf220 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_244], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (768, 256), (1, 768), 0), view_58, out=buf220)
        del buf219
        del view_58
        buf224 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_248], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (3072, 256), (1, 3072), 0), view_56, out=buf224)
        del buf223
        del view_56
        buf250 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_264], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (768, 256), (1, 768), 0), view_46, out=buf250)
        del buf249
        del view_46
        buf254 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_268], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (3072, 256), (1, 3072), 0), view_44, out=buf254)
        del buf253
        del view_44
        buf280 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_284], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (768, 256), (1, 768), 0), view_34, out=buf280)
        del buf279
        del view_34
        buf284 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_288], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (3072, 256), (1, 3072), 0), view_32, out=buf284)
        del buf283
        del view_32
        buf310 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_304], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (768, 256), (1, 768), 0), view_22, out=buf310)
        del buf309
        del view_22
        buf314 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_308], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (3072, 256), (1, 3072), 0), view_20, out=buf314)
        del buf313
        del view_20
        buf340 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_324], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (768, 256), (1, 768), 0), view_10, out=buf340)
        del buf339
        del view_10
        buf344 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_328], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (3072, 256), (1, 3072), 0), view_8, out=buf344)
        del buf343
        del view_8
        buf33 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf31, buf33, 1769472, stream=stream0)
        del buf31
        buf63 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf61, buf63, 1769472, stream=stream0)
        del buf61
        buf93 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf91, buf93, 1769472, stream=stream0)
        del buf91
        buf123 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf121, buf123, 1769472, stream=stream0)
        del buf121
        buf153 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf151, buf153, 1769472, stream=stream0)
        del buf151
        buf183 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf181, buf183, 1769472, stream=stream0)
        del buf181
        buf213 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf211, buf213, 1769472, stream=stream0)
        del buf211
        buf243 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf241, buf243, 1769472, stream=stream0)
        del buf241
        buf273 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf271, buf273, 1769472, stream=stream0)
        del buf271
        buf303 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf301, buf303, 1769472, stream=stream0)
        del buf301
        buf333 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf331, buf333, 1769472, stream=stream0)
        del buf331
        buf363 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf361, buf363, 1769472, stream=stream0)
        del buf361
        buf12 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf10, buf12, 2359296, stream=stream0)
        del buf10
        buf16 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf14, buf16, 2359296, stream=stream0)
        del buf14
        buf42 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf40, buf42, 2359296, stream=stream0)
        del buf40
        buf46 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf44, buf46, 2359296, stream=stream0)
        del buf44
        buf72 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf70, buf72, 2359296, stream=stream0)
        del buf70
        buf76 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf74, buf76, 2359296, stream=stream0)
        del buf74
        buf102 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf100, buf102, 2359296, stream=stream0)
        del buf100
        buf106 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf104, buf106, 2359296, stream=stream0)
        del buf104
        buf132 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf130, buf132, 2359296, stream=stream0)
        del buf130
        buf136 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf134, buf136, 2359296, stream=stream0)
        del buf134
        buf162 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf160, buf162, 2359296, stream=stream0)
        del buf160
        buf166 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf164, buf166, 2359296, stream=stream0)
        del buf164
        buf192 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf190, buf192, 2359296, stream=stream0)
        del buf190
        buf196 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf194, buf196, 2359296, stream=stream0)
        del buf194
        buf222 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf220, buf222, 2359296, stream=stream0)
        del buf220
        buf226 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf224, buf226, 2359296, stream=stream0)
        del buf224
        buf252 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf250, buf252, 2359296, stream=stream0)
        del buf250
        buf256 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf254, buf256, 2359296, stream=stream0)
        del buf254
        buf282 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf280, buf282, 2359296, stream=stream0)
        del buf280
        buf286 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf284, buf286, 2359296, stream=stream0)
        del buf284
        buf312 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf310, buf312, 2359296, stream=stream0)
        del buf310
        buf316 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf314, buf316, 2359296, stream=stream0)
        del buf314
        buf342 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf340, buf342, 2359296, stream=stream0)
        del buf340
        buf346 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(buf344, buf346, 2359296, stream=stream0)
        del buf344
    return (None, buf373, buf369, buf367, buf363, buf355, buf350, buf346, buf342, buf337, buf333, buf325, buf320, buf316, buf312, buf307, buf303, buf295, buf290, buf286, buf282, buf277, buf273, buf265, buf260, buf256, buf252, buf247, buf243, buf235, buf230, buf226, buf222, buf217, buf213, buf205, buf200, buf196, buf192, buf187, buf183, buf175, buf170, buf166, buf162, buf157, buf153, buf145, buf140, buf136, buf132, buf127, buf123, buf115, buf110, buf106, buf102, buf97, buf93, buf85, buf80, buf76, buf72, buf67, buf63, buf55, buf50, buf46, buf42, buf37, buf33, buf25, buf20, buf16, buf12, buf8, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    iota = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    embedding_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_3 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_6 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_12 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_2 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_2 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_10 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_7 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_9 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_10 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_11 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_22 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_28 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_9 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_6 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_22 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_14 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_17 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_18 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_19 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_38 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_44 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_16 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_10 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_34 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_21 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_25 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_26 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_27 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_54 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_60 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_23 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_14 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_46 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_28 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_33 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_34 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_35 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_69 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_70 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_76 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_30 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_58 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_35 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_41 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_42 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_43 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_85 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_86 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_92 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_37 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_70 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_42 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_49 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_50 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_51 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_101 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_102 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_108 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_44 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_26 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_82 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_49 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_57 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_58 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_59 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_117 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_118 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_124 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_51 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_30 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_94 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_56 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_96 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_65 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_66 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_67 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_133 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_134 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_140 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_58 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_34 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_106 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_63 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_73 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_74 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_75 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_149 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_150 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_156 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_65 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_38 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_118 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_70 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_81 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_82 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_83 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_165 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_166 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_172 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_72 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_42 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_130 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_77 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_89 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_90 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_91 = rand_strided((4, 12, 64, 64), (147456, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_181 = rand_strided((4, 12, 64, 64), (49152, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_182 = rand_strided((4, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_187 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_188 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mul_79 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_46 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_142 = rand_strided((256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_84 = rand_strided((4, 64, 768), (49152, 768, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_145 = rand_strided((4, 64, 65), (4160, 65, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((256, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((256, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_199 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_99 = rand_strided((65, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_2 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_107 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_3 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_111 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_119 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_4 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_127 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_5 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_139 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_6 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_147 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_7 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_159 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_8 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_167 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_9 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_179 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_10 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_187 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_11 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_199 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_12 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_207 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_13 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_219 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_14 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_227 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_15 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_239 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_16 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_243 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_247 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_17 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_259 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_18 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_267 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_19 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_271 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_279 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_20 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_287 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_21 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_299 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_22 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_307 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_23 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_319 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_24 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_327 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    div_25 = rand_strided((4, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_339 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((4, 64, 65), (4160, 65, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_4, primals_7, primals_10, primals_13, primals_16, primals_19, primals_22, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_77, iota, embedding, embedding_1, getitem_1, rsqrt, view, permute_1, permute_2, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mul_2, view_8, mm_2, view_10, mul_7, view_12, permute_9, permute_10, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_9, view_20, mm_6, view_22, mul_14, view_24, permute_17, permute_18, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_16, view_32, mm_10, view_34, mul_21, view_36, permute_25, permute_26, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_23, view_44, mm_14, view_46, mul_28, view_48, permute_33, permute_34, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_30, view_56, mm_18, view_58, mul_35, view_60, permute_41, permute_42, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_37, view_68, mm_22, view_70, mul_42, view_72, permute_49, permute_50, permute_51, getitem_101, getitem_102, getitem_107, getitem_108, mul_44, view_80, mm_26, view_82, mul_49, view_84, permute_57, permute_58, permute_59, getitem_117, getitem_118, getitem_123, getitem_124, mul_51, view_92, mm_30, view_94, mul_56, view_96, permute_65, permute_66, permute_67, getitem_133, getitem_134, getitem_139, getitem_140, mul_58, view_104, mm_34, view_106, mul_63, view_108, permute_73, permute_74, permute_75, getitem_149, getitem_150, getitem_155, getitem_156, mul_65, view_116, mm_38, view_118, mul_70, view_120, permute_81, permute_82, permute_83, getitem_165, getitem_166, getitem_171, getitem_172, mul_72, view_128, mm_42, view_130, mul_77, view_132, permute_89, permute_90, permute_91, getitem_181, getitem_182, getitem_187, getitem_188, mul_79, view_140, mm_46, view_142, mul_84, view_144, view_145, amax, log, convert_element_type_199, permute_99, div_2, permute_103, permute_107, div_3, permute_111, permute_119, div_4, permute_123, permute_127, div_5, permute_131, permute_139, div_6, permute_143, permute_147, div_7, permute_151, permute_159, div_8, permute_163, permute_167, div_9, permute_171, permute_179, div_10, permute_183, permute_187, div_11, permute_191, permute_199, div_12, permute_203, permute_207, div_13, permute_211, permute_219, div_14, permute_223, permute_227, div_15, permute_231, permute_239, div_16, permute_243, permute_247, div_17, permute_251, permute_259, div_18, permute_263, permute_267, div_19, permute_271, permute_279, div_20, permute_283, permute_287, div_21, permute_291, permute_299, div_22, permute_303, permute_307, div_23, permute_311, permute_319, div_24, permute_323, permute_327, div_25, permute_331, permute_339, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
