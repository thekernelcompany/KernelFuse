# AOT ID: ['1_forward']
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


# kernel path: /tmp/torchinductor_ubuntu/jo/cjolfbha5otiklkoiqdgkqtedy5ywsrne5zwscusewphtg2ybdl2.py
# Topologically Sorted Source Nodes: [pos], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   pos => iota
# Graph fragment:
#   %iota : [num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_0 = async_compile.triton('triton_poi_fused_arange_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/4z/c4zb7fnfhdhh3wr7h7r37tople7txwybrkcb6vtxgesgsr3yzi4q.py
# Topologically Sorted Source Nodes: [pos_emb], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   pos_emb => embedding_1
# Graph fragment:
#   %embedding_1 : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %iota), kwargs = {})
triton_poi_fused_embedding_1 = async_compile.triton('triton_poi_fused_embedding_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/3b/c3bdpe7am6r2yp6jnjy36jxzi27w3xc65ivdwpchugr3zwmysz4f.py
# Topologically Sorted Source Nodes: [tok_emb, add, layer_norm, linear], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   layer_norm => add_1, mul, mul_1, rsqrt, sub, var_mean
#   linear => convert_element_type_1
#   tok_emb => embedding
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_4), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_embedding_native_layer_norm_2 = async_compile.triton('triton_per_fused__to_copy_add_embedding_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_native_layer_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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
    x0 = xindex
    r0_1 = r0_index
    x2 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 65), "index out of bounds: 0 <= tmp4 < 65")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 768.0
    tmp26 = (tmp24 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp8 - tmp18
    tmp31 = tmp30 * tmp29
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp6, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, None)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp34, r0_mask)
    tl.store(out_ptr1 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/qi/cqiyxqepgfjdb7qwy27vit2ab6vwy7whqyxrtwtohy3cjb5uwoea.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type, permute
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_3 = async_compile.triton('triton_poi_fused__to_copy_t_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/pv/cpv4h4p7yrvhpgnso3p3cnbehtqrbondp5lwx4xrvw7zddthxobm.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_4, permute_5
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.bfloat16), kwargs = {})
#   %permute_5 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_4, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_4 = async_compile.triton('triton_poi_fused__to_copy_t_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/67/c67s7zanuwi7q3zk2h6bkc6tuoamx77luvk7jfjy3f5srrssjru4.py
# Topologically Sorted Source Nodes: [add, x_1, layer_norm_1, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   layer_norm_1 => add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_1 => add_2
#   x_2 => convert_element_type_8
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_7), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_15), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_7), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp5 - tmp15
    tmp23 = 768.0
    tmp24 = (tmp21 / tmp23)
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 0.0013020833333333333
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp28, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp31, r0_mask)
    tl.store(out_ptr4 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/od/coda6wkafch5aoavzw7o3tny26ttwtxkwr2ds3cub5alspavmcfs.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   x_2 => convert_element_type_7, permute_6
# Graph fragment:
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %permute_6 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_7, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_6 = async_compile.triton('triton_poi_fused__to_copy_t_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/56/c56fyi56qk33a67nnqxjanl64qmu6jwttds26yzy25ozoiedr7gb.py
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
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
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
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/gk/cgkn6kf4afmcac7s2jirruqra7adar7wlgm3wmsc5o5xzduzyll4.py
# Topologically Sorted Source Nodes: [add, x_1, x_6, layer_norm_2, linear_4], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   layer_norm_2 => add_6, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
#   linear_4 => convert_element_type_17
#   x_1 => add_2
#   x_6 => add_5
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_7), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_11), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_17), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_10), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = (tmp24 / tmp26)
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 0.0013020833333333333
    tmp36 = tmp30 * tmp35
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp34, r0_mask)
    tl.store(out_ptr4 + (x3), tmp36, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/pr/cpr45kqypnfxmcmjme4oeaklfktjen7zgmo4nyufy4zj7tiqdfwc.py
# Topologically Sorted Source Nodes: [add, x_1, x_6, x_7, layer_norm_3, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   layer_norm_3 => add_8, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_1 => add_2
#   x_6 => add_5
#   x_7 => add_7
#   x_8 => convert_element_type_24
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_7), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_11), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_19), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_31), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_13), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
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
    tmp38 = 0.0013020833333333333
    tmp39 = tmp33 * tmp38
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp11, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp34, r0_mask)
    tl.store(out_ptr4 + (r0_2 + 768*x3), tmp37, r0_mask)
    tl.store(out_ptr5 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/j4/cj4fdx5k5sffdxxeok7xa7nunv2djw34nub6lnpordpya5mauly3.py
# Topologically Sorted Source Nodes: [x_12, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_4 => add_11, mul_14, mul_15, rsqrt_4, sub_4, var_mean_4
#   linear_8 => convert_element_type_33
#   x_12 => add_10
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_33), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %primals_16), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    tmp30 = 0.0013020833333333333
    tmp31 = tmp25 * tmp30
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp26, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp29, r0_mask)
    tl.store(out_ptr4 + (x0), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/fu/cfuhc6jgjzqvensbsxqevhnfdbfe3fro2x3tzfjierbc4mjkf4em.py
# Topologically Sorted Source Nodes: [x_12, x_13, layer_norm_5, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_5 => add_13, mul_16, mul_17, rsqrt_5, sub_5, var_mean_5
#   x_12 => add_10
#   x_13 => add_12
#   x_14 => convert_element_type_40
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_31), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_47), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %primals_19), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    tmp33 = 0.0013020833333333333
    tmp34 = tmp28 * tmp33
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp29, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp32, r0_mask)
    tl.store(out_ptr4 + (x0), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/yq/cyqzqm7kudihjsu5i4fctctsz3hh62tlvp5ji4jqsghyvehi4wq3.py
# Topologically Sorted Source Nodes: [x_12, x_13, x_18, layer_norm_6, linear_12], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_6 => add_16, mul_21, mul_22, rsqrt_6, sub_6, var_mean_6
#   linear_12 => convert_element_type_49
#   x_12 => add_10
#   x_13 => add_12
#   x_18 => add_15
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_31), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_35), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_49), kwargs = {})
#   %mul_21 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %primals_22), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_6, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    tmp36 = 0.0013020833333333333
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/yj/cyjebowh35uu4nkk6olzgit3o7e4olbturleudrsuu4fxvoh3bam.py
# Topologically Sorted Source Nodes: [x_12, x_13, x_18, x_19, layer_norm_7, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_7 => add_18, mul_23, mul_24, rsqrt_7, sub_7, var_mean_7
#   x_12 => add_10
#   x_13 => add_12
#   x_18 => add_15
#   x_19 => add_17
#   x_20 => convert_element_type_56
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_23), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_31), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_35), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_43), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_17, %getitem_63), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %primals_25), kwargs = {})
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    tmp39 = 0.0013020833333333333
    tmp40 = tmp34 * tmp39
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp38, r0_mask)
    tl.store(out_ptr4 + (x0), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/db/cdbwxpfckugdxpfbhjf5ndc6h6lb62cu7n4gqtj5ounxd7ts7s3f.py
# Topologically Sorted Source Nodes: [x_72, x_73, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   logits => convert_element_type_193
#   x_72 => add_60
#   x_73 => add_61, mul_84, mul_85, rsqrt_24, sub_24, var_mean_24
# Graph fragment:
#   %add_60 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %view_143), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_60, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_61,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_60, %getitem_193), kwargs = {})
#   %mul_84 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %primals_76), kwargs = {})
#   %convert_element_type_193 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_85, torch.bfloat16), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_24, 768), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_14(in_out_ptr0, in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
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
    tmp27 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tmp30 = 0.0013020833333333333
    tmp31 = tmp25 * tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp29, r0_mask)
    tl.store(out_ptr3 + (x0), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/eq/ceqdf7ketna7rcjplvfv2g6gxwdq5ypzwqi3xeeikhsff7ch7pqa.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   logits => convert_element_type_192, permute_96
# Graph fragment:
#   %convert_element_type_192 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   %permute_96 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_192, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_15 = async_compile.triton('triton_poi_fused__to_copy_t_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/yf/cyfsebg5b3hcrot4zziibi7w7643p46pnryfktyspkdzja2h3tix.py
# Topologically Sorted Source Nodes: [loss, ], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   loss => convert_element_type_196, log
# Graph fragment:
#   %convert_element_type_196 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_146, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_196, 1), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_195,), kwargs = {})
triton_per_fused__log_softmax_prepare_softmax_online_16 = async_compile.triton('triton_per_fused__log_softmax_prepare_softmax_online_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_prepare_softmax_online_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_prepare_softmax_online_16(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp14 = tl_math.log(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ubuntu/af/cafzytsofksuxmd22jkzaiwat2mlyel2vgsrghqojnxh2xi7bfpw.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_199, div, full_default_1, ne, neg, sum_2, sum_3, where_1
# Graph fragment:
#   %ne : [num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_147, -1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_199 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_199), kwargs = {})
triton_per_fused_nll_loss_forward_17 = async_compile.triton('triton_per_fused_nll_loss_forward_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'DB9AB38F8EC6F0345E7993EA6F0B65F5D226C6A85B926CA1119304BCEC62BF55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_nll_loss_forward_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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
    tmp16 = tl.load(in_ptr2 + (r0_0), None)
    tmp18 = tl.load(in_ptr3 + (r0_0), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = tl.where(tmp2, tmp0, tmp7)
    tmp9 = tl.full([R0_BLOCK], 65, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert((0 <= tmp12) & (tmp12 < 65), "index out of bounds: 0 <= tmp12 < 65")
    tmp14 = tl.load(in_ptr1 + (tmp12 + 65*r0_0), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 - tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = -tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tmp6.to(tl.float32)
    tmp29 = (tmp27 / tmp28)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp28, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp29, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77 = args
    args.clear()
    assert_size_stride(primals_1, (4, 64), (64, 1))
    assert_size_stride(primals_2, (65, 768), (768, 1))
    assert_size_stride(primals_3, (64, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (2304, 768), (768, 1))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (3072, 768), (768, 1))
    assert_size_stride(primals_9, (768, 3072), (3072, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (2304, 768), (768, 1))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (3072, 768), (768, 1))
    assert_size_stride(primals_15, (768, 3072), (3072, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (2304, 768), (768, 1))
    assert_size_stride(primals_18, (768, 768), (768, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (3072, 768), (768, 1))
    assert_size_stride(primals_21, (768, 3072), (3072, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (2304, 768), (768, 1))
    assert_size_stride(primals_24, (768, 768), (768, 1))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (3072, 768), (768, 1))
    assert_size_stride(primals_27, (768, 3072), (3072, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (2304, 768), (768, 1))
    assert_size_stride(primals_30, (768, 768), (768, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (3072, 768), (768, 1))
    assert_size_stride(primals_33, (768, 3072), (3072, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (2304, 768), (768, 1))
    assert_size_stride(primals_36, (768, 768), (768, 1))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (3072, 768), (768, 1))
    assert_size_stride(primals_39, (768, 3072), (3072, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (2304, 768), (768, 1))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (3072, 768), (768, 1))
    assert_size_stride(primals_45, (768, 3072), (3072, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (2304, 768), (768, 1))
    assert_size_stride(primals_48, (768, 768), (768, 1))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (3072, 768), (768, 1))
    assert_size_stride(primals_51, (768, 3072), (3072, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (2304, 768), (768, 1))
    assert_size_stride(primals_54, (768, 768), (768, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (3072, 768), (768, 1))
    assert_size_stride(primals_57, (768, 3072), (3072, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (2304, 768), (768, 1))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (3072, 768), (768, 1))
    assert_size_stride(primals_63, (768, 3072), (3072, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (2304, 768), (768, 1))
    assert_size_stride(primals_66, (768, 768), (768, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (3072, 768), (768, 1))
    assert_size_stride(primals_69, (768, 3072), (3072, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (2304, 768), (768, 1))
    assert_size_stride(primals_72, (768, 768), (768, 1))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (3072, 768), (768, 1))
    assert_size_stride(primals_75, (768, 3072), (3072, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (4, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_0.run(buf0, 64, stream=stream0)
        buf2 = empty_strided_cuda((64, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pos_emb], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_1.run(primals_3, buf2, 49152, stream=stream0)
        del primals_3
        buf1 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 64, 1), (64, 1, 256), torch.float32)
        buf6 = reinterpret_tensor(buf4, (4, 64, 1), (64, 1, 1), 0); del buf4  # reuse
        buf8 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tok_emb, add, layer_norm, linear], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_native_layer_norm_2.run(buf6, primals_1, primals_2, buf2, primals_4, buf1, buf3, buf8, 256, 768, stream=stream0)
        buf7 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_5, buf7, 1769472, stream=stream0)
        del primals_5
        buf9 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (256, 768), (768, 1), 0), buf7, out=buf9)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf10 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf11 = buf10[0]
        assert_size_stride(buf11, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf12 = buf10[1]
        assert_size_stride(buf12, (4, 12, 64), (768, 64, 1))
        buf13 = buf10[6]
        assert_size_stride(buf13, (2, ), (1, ))
        buf14 = buf10[7]
        assert_size_stride(buf14, (), ())
        del buf10
        buf16 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_6, buf16, 589824, stream=stream0)
        del primals_6
        buf17 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (256, 768), (768, 1), 0), buf16, out=buf17)
        buf21 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf23 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf345 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_1, layer_norm_1, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_5.run(buf1, buf2, buf17, primals_7, buf21, buf23, buf345, 256, 768, stream=stream0)
        buf22 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_8, buf22, 2359296, stream=stream0)
        del primals_8
        buf24 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (256, 768), (768, 1), 0), buf22, out=buf24)
        buf25 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_9, buf25, 2359296, stream=stream0)
        del primals_9
        buf26 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf24, buf26, 786432, stream=stream0)
        buf27 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (256, 3072), (3072, 1), 0), buf25, out=buf27)
        buf31 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf344 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_1, x_6, layer_norm_2, linear_4], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8.run(buf1, buf2, buf17, buf27, primals_10, buf31, buf33, buf344, 256, 768, stream=stream0)
        buf32 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_11, buf32, 1769472, stream=stream0)
        del primals_11
        buf34 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (256, 768), (768, 1), 0), buf32, out=buf34)
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf35 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf36 = buf35[0]
        assert_size_stride(buf36, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf37 = buf35[1]
        assert_size_stride(buf37, (4, 12, 64), (768, 64, 1))
        buf38 = buf35[6]
        assert_size_stride(buf38, (2, ), (1, ))
        buf39 = buf35[7]
        assert_size_stride(buf39, (), ())
        del buf35
        buf41 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_12, buf41, 589824, stream=stream0)
        del primals_12
        buf42 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (256, 768), (768, 1), 0), buf41, out=buf42)
        buf43 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf47 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf49 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf343 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_1, x_6, x_7, layer_norm_3, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9.run(buf1, buf2, buf17, buf27, buf42, primals_13, buf43, buf47, buf49, buf343, 256, 768, stream=stream0)
        buf48 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_14, buf48, 2359296, stream=stream0)
        del primals_14
        buf50 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (256, 768), (768, 1), 0), buf48, out=buf50)
        buf51 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_15, buf51, 2359296, stream=stream0)
        del primals_15
        buf52 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf50, buf52, 786432, stream=stream0)
        buf53 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (256, 3072), (3072, 1), 0), buf51, out=buf53)
        buf57 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf59 = reinterpret_tensor(buf27, (4, 64, 768), (49152, 768, 1), 0); del buf27  # reuse
        buf342 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf43, buf53, primals_16, buf57, buf59, buf342, 256, 768, stream=stream0)
        buf58 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_17, buf58, 1769472, stream=stream0)
        del primals_17
        buf60 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (256, 768), (768, 1), 0), buf58, out=buf60)
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf61 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf62 = buf61[0]
        assert_size_stride(buf62, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf63 = buf61[1]
        assert_size_stride(buf63, (4, 12, 64), (768, 64, 1))
        buf64 = buf61[6]
        assert_size_stride(buf64, (2, ), (1, ))
        buf65 = buf61[7]
        assert_size_stride(buf65, (), ())
        del buf61
        buf67 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_18, buf67, 589824, stream=stream0)
        del primals_18
        buf68 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (256, 768), (768, 1), 0), buf67, out=buf68)
        buf72 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf74 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf341 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, layer_norm_5, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf43, buf53, buf68, primals_19, buf72, buf74, buf341, 256, 768, stream=stream0)
        buf73 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_20, buf73, 2359296, stream=stream0)
        del primals_20
        buf75 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (256, 768), (768, 1), 0), buf73, out=buf75)
        buf76 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_21, buf76, 2359296, stream=stream0)
        del primals_21
        buf77 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf75, buf77, 786432, stream=stream0)
        buf78 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (256, 3072), (3072, 1), 0), buf76, out=buf78)
        buf82 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf84 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf340 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, x_18, layer_norm_6, linear_12], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf43, buf53, buf68, buf78, primals_22, buf82, buf84, buf340, 256, 768, stream=stream0)
        buf83 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_23, buf83, 1769472, stream=stream0)
        del primals_23
        buf85 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (256, 768), (768, 1), 0), buf83, out=buf85)
        # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf86 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf87 = buf86[0]
        assert_size_stride(buf87, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf88 = buf86[1]
        assert_size_stride(buf88, (4, 12, 64), (768, 64, 1))
        buf89 = buf86[6]
        assert_size_stride(buf89, (2, ), (1, ))
        buf90 = buf86[7]
        assert_size_stride(buf90, (), ())
        del buf86
        buf92 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_24, buf92, 589824, stream=stream0)
        del primals_24
        buf93 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (256, 768), (768, 1), 0), buf92, out=buf93)
        buf94 = buf43; del buf43  # reuse
        buf98 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf100 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf339 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, x_18, x_19, layer_norm_7, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf94, buf53, buf68, buf78, buf93, primals_25, buf98, buf100, buf339, 256, 768, stream=stream0)
        buf99 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_26, buf99, 2359296, stream=stream0)
        del primals_26
        buf101 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (256, 768), (768, 1), 0), buf99, out=buf101)
        buf102 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_27, buf102, 2359296, stream=stream0)
        del primals_27
        buf103 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf101, buf103, 786432, stream=stream0)
        buf104 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (256, 3072), (3072, 1), 0), buf102, out=buf104)
        buf108 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf110 = reinterpret_tensor(buf78, (4, 64, 768), (49152, 768, 1), 0); del buf78  # reuse
        buf338 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24, layer_norm_8, linear_16], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf94, buf104, primals_28, buf108, buf110, buf338, 256, 768, stream=stream0)
        buf109 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_29, buf109, 1769472, stream=stream0)
        del primals_29
        buf111 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (256, 768), (768, 1), 0), buf109, out=buf111)
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf112 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf113 = buf112[0]
        assert_size_stride(buf113, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf114 = buf112[1]
        assert_size_stride(buf114, (4, 12, 64), (768, 64, 1))
        buf115 = buf112[6]
        assert_size_stride(buf115, (2, ), (1, ))
        buf116 = buf112[7]
        assert_size_stride(buf116, (), ())
        del buf112
        buf118 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_30, buf118, 589824, stream=stream0)
        del primals_30
        buf119 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (256, 768), (768, 1), 0), buf118, out=buf119)
        buf123 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf125 = reinterpret_tensor(buf53, (4, 64, 768), (49152, 768, 1), 0); del buf53  # reuse
        buf337 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24, x_25, layer_norm_9, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf94, buf104, buf119, primals_31, buf123, buf125, buf337, 256, 768, stream=stream0)
        buf124 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_32, buf124, 2359296, stream=stream0)
        del primals_32
        buf126 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (256, 768), (768, 1), 0), buf124, out=buf126)
        buf127 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_33, buf127, 2359296, stream=stream0)
        del primals_33
        buf128 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf126, buf128, 786432, stream=stream0)
        buf129 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (256, 3072), (3072, 1), 0), buf127, out=buf129)
        buf133 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf135 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf336 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24, x_25, x_30, layer_norm_10, linear_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf94, buf104, buf119, buf129, primals_34, buf133, buf135, buf336, 256, 768, stream=stream0)
        buf134 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_35, buf134, 1769472, stream=stream0)
        del primals_35
        buf136 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (256, 768), (768, 1), 0), buf134, out=buf136)
        # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf137 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf138 = buf137[0]
        assert_size_stride(buf138, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf139 = buf137[1]
        assert_size_stride(buf139, (4, 12, 64), (768, 64, 1))
        buf140 = buf137[6]
        assert_size_stride(buf140, (2, ), (1, ))
        buf141 = buf137[7]
        assert_size_stride(buf141, (), ())
        del buf137
        buf143 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_36, buf143, 589824, stream=stream0)
        del primals_36
        buf144 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (256, 768), (768, 1), 0), buf143, out=buf144)
        buf145 = buf94; del buf94  # reuse
        buf149 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf151 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf335 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24, x_25, x_30, x_31, layer_norm_11, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf145, buf104, buf119, buf129, buf144, primals_37, buf149, buf151, buf335, 256, 768, stream=stream0)
        buf150 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_38, buf150, 2359296, stream=stream0)
        del primals_38
        buf152 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (256, 768), (768, 1), 0), buf150, out=buf152)
        buf153 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_39, buf153, 2359296, stream=stream0)
        del primals_39
        buf154 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf152, buf154, 786432, stream=stream0)
        buf155 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (256, 3072), (3072, 1), 0), buf153, out=buf155)
        buf159 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf161 = reinterpret_tensor(buf129, (4, 64, 768), (49152, 768, 1), 0); del buf129  # reuse
        buf334 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, layer_norm_12, linear_24], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf145, buf155, primals_40, buf159, buf161, buf334, 256, 768, stream=stream0)
        buf160 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_41, buf160, 1769472, stream=stream0)
        del primals_41
        buf162 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (256, 768), (768, 1), 0), buf160, out=buf162)
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf163 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf164 = buf163[0]
        assert_size_stride(buf164, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf165 = buf163[1]
        assert_size_stride(buf165, (4, 12, 64), (768, 64, 1))
        buf166 = buf163[6]
        assert_size_stride(buf166, (2, ), (1, ))
        buf167 = buf163[7]
        assert_size_stride(buf167, (), ())
        del buf163
        buf169 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_42, buf169, 589824, stream=stream0)
        del primals_42
        buf170 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (256, 768), (768, 1), 0), buf169, out=buf170)
        buf174 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf176 = reinterpret_tensor(buf104, (4, 64, 768), (49152, 768, 1), 0); del buf104  # reuse
        buf333 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, x_37, layer_norm_13, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf145, buf155, buf170, primals_43, buf174, buf176, buf333, 256, 768, stream=stream0)
        buf175 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_44, buf175, 2359296, stream=stream0)
        del primals_44
        buf177 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (256, 768), (768, 1), 0), buf175, out=buf177)
        buf178 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_45, buf178, 2359296, stream=stream0)
        del primals_45
        buf179 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf177, buf179, 786432, stream=stream0)
        buf180 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (256, 3072), (3072, 1), 0), buf178, out=buf180)
        buf184 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf186 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf332 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, x_37, x_42, layer_norm_14, linear_28], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf145, buf155, buf170, buf180, primals_46, buf184, buf186, buf332, 256, 768, stream=stream0)
        buf185 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_47, buf185, 1769472, stream=stream0)
        del primals_47
        buf187 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (256, 768), (768, 1), 0), buf185, out=buf187)
        # Topologically Sorted Source Nodes: [y_21], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf188 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf189 = buf188[0]
        assert_size_stride(buf189, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf190 = buf188[1]
        assert_size_stride(buf190, (4, 12, 64), (768, 64, 1))
        buf191 = buf188[6]
        assert_size_stride(buf191, (2, ), (1, ))
        buf192 = buf188[7]
        assert_size_stride(buf192, (), ())
        del buf188
        buf194 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_48, buf194, 589824, stream=stream0)
        del primals_48
        buf195 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (256, 768), (768, 1), 0), buf194, out=buf195)
        buf196 = buf145; del buf145  # reuse
        buf200 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf202 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf331 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, x_37, x_42, x_43, layer_norm_15, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf196, buf155, buf170, buf180, buf195, primals_49, buf200, buf202, buf331, 256, 768, stream=stream0)
        buf201 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_50, buf201, 2359296, stream=stream0)
        del primals_50
        buf203 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (256, 768), (768, 1), 0), buf201, out=buf203)
        buf204 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_51, buf204, 2359296, stream=stream0)
        del primals_51
        buf205 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf203, buf205, 786432, stream=stream0)
        buf206 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (256, 3072), (3072, 1), 0), buf204, out=buf206)
        buf210 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf212 = reinterpret_tensor(buf180, (4, 64, 768), (49152, 768, 1), 0); del buf180  # reuse
        buf330 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, layer_norm_16, linear_32], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf196, buf206, primals_52, buf210, buf212, buf330, 256, 768, stream=stream0)
        buf211 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_53, buf211, 1769472, stream=stream0)
        del primals_53
        buf213 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (256, 768), (768, 1), 0), buf211, out=buf213)
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf214 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf215 = buf214[0]
        assert_size_stride(buf215, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf216 = buf214[1]
        assert_size_stride(buf216, (4, 12, 64), (768, 64, 1))
        buf217 = buf214[6]
        assert_size_stride(buf217, (2, ), (1, ))
        buf218 = buf214[7]
        assert_size_stride(buf218, (), ())
        del buf214
        buf220 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_54, buf220, 589824, stream=stream0)
        del primals_54
        buf221 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (256, 768), (768, 1), 0), buf220, out=buf221)
        buf225 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf227 = reinterpret_tensor(buf155, (4, 64, 768), (49152, 768, 1), 0); del buf155  # reuse
        buf329 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_49, layer_norm_17, x_50], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf196, buf206, buf221, primals_55, buf225, buf227, buf329, 256, 768, stream=stream0)
        buf226 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_56, buf226, 2359296, stream=stream0)
        del primals_56
        buf228 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (256, 768), (768, 1), 0), buf226, out=buf228)
        buf229 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_57, buf229, 2359296, stream=stream0)
        del primals_57
        buf230 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf228, buf230, 786432, stream=stream0)
        buf231 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf230, (256, 3072), (3072, 1), 0), buf229, out=buf231)
        buf235 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf237 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf328 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_49, x_54, layer_norm_18, linear_36], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf196, buf206, buf221, buf231, primals_58, buf235, buf237, buf328, 256, 768, stream=stream0)
        buf236 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_59, buf236, 1769472, stream=stream0)
        del primals_59
        buf238 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (256, 768), (768, 1), 0), buf236, out=buf238)
        # Topologically Sorted Source Nodes: [y_27], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf239 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf240 = buf239[0]
        assert_size_stride(buf240, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf241 = buf239[1]
        assert_size_stride(buf241, (4, 12, 64), (768, 64, 1))
        buf242 = buf239[6]
        assert_size_stride(buf242, (2, ), (1, ))
        buf243 = buf239[7]
        assert_size_stride(buf243, (), ())
        del buf239
        buf245 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_60, buf245, 589824, stream=stream0)
        del primals_60
        buf246 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (256, 768), (768, 1), 0), buf245, out=buf246)
        buf247 = buf196; del buf196  # reuse
        buf251 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf253 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf327 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_49, x_54, x_55, layer_norm_19, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf247, buf206, buf221, buf231, buf246, primals_61, buf251, buf253, buf327, 256, 768, stream=stream0)
        buf252 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_62, buf252, 2359296, stream=stream0)
        del primals_62
        buf254 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (256, 768), (768, 1), 0), buf252, out=buf254)
        buf255 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_63, buf255, 2359296, stream=stream0)
        del primals_63
        buf256 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf254, buf256, 786432, stream=stream0)
        buf257 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (256, 3072), (3072, 1), 0), buf255, out=buf257)
        buf261 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf263 = reinterpret_tensor(buf231, (4, 64, 768), (49152, 768, 1), 0); del buf231  # reuse
        buf326 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, layer_norm_20, linear_40], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf247, buf257, primals_64, buf261, buf263, buf326, 256, 768, stream=stream0)
        buf262 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_65, buf262, 1769472, stream=stream0)
        del primals_65
        buf264 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (256, 768), (768, 1), 0), buf262, out=buf264)
        # Topologically Sorted Source Nodes: [y_30], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf265 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf266 = buf265[0]
        assert_size_stride(buf266, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf267 = buf265[1]
        assert_size_stride(buf267, (4, 12, 64), (768, 64, 1))
        buf268 = buf265[6]
        assert_size_stride(buf268, (2, ), (1, ))
        buf269 = buf265[7]
        assert_size_stride(buf269, (), ())
        del buf265
        buf271 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_66, buf271, 589824, stream=stream0)
        del primals_66
        buf272 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (256, 768), (768, 1), 0), buf271, out=buf272)
        buf276 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf278 = reinterpret_tensor(buf206, (4, 64, 768), (49152, 768, 1), 0); del buf206  # reuse
        buf325 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61, layer_norm_21, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf247, buf257, buf272, primals_67, buf276, buf278, buf325, 256, 768, stream=stream0)
        buf277 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_68, buf277, 2359296, stream=stream0)
        del primals_68
        buf279 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (256, 768), (768, 1), 0), buf277, out=buf279)
        buf280 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_69, buf280, 2359296, stream=stream0)
        del primals_69
        buf281 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf279, buf281, 786432, stream=stream0)
        buf282 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (256, 3072), (3072, 1), 0), buf280, out=buf282)
        buf286 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf288 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf324 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61, x_66, layer_norm_22, linear_44], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf247, buf257, buf272, buf282, primals_70, buf286, buf288, buf324, 256, 768, stream=stream0)
        buf287 = empty_strided_cuda((768, 2304), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_3.run(primals_71, buf287, 1769472, stream=stream0)
        del primals_71
        buf289 = empty_strided_cuda((256, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (256, 768), (768, 1), 0), buf287, out=buf289)
        # Topologically Sorted Source Nodes: [y_33], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf290 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        buf291 = buf290[0]
        assert_size_stride(buf291, (4, 12, 64, 64), (49152, 64, 768, 1))
        buf292 = buf290[1]
        assert_size_stride(buf292, (4, 12, 64), (768, 64, 1))
        buf293 = buf290[6]
        assert_size_stride(buf293, (2, ), (1, ))
        buf294 = buf290[7]
        assert_size_stride(buf294, (), ())
        del buf290
        buf296 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_72, buf296, 589824, stream=stream0)
        del primals_72
        buf297 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (256, 768), (768, 1), 0), buf296, out=buf297)
        buf298 = buf247; del buf247  # reuse
        buf302 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.float32)
        buf304 = empty_strided_cuda((4, 64, 768), (49152, 768, 1), torch.bfloat16)
        buf323 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61, x_66, x_67, layer_norm_23, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf298, buf257, buf272, buf282, buf297, primals_73, buf302, buf304, buf323, 256, 768, stream=stream0)
        del buf257
        del buf272
        buf303 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_74, buf303, 2359296, stream=stream0)
        del primals_74
        buf305 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (256, 768), (768, 1), 0), buf303, out=buf305)
        buf306 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_75, buf306, 2359296, stream=stream0)
        del primals_75
        buf307 = empty_strided_cuda((4, 64, 3072), (196608, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf305, buf307, 786432, stream=stream0)
        buf308 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (256, 3072), (3072, 1), 0), buf306, out=buf308)
        buf312 = buf298; del buf298  # reuse
        buf314 = reinterpret_tensor(buf282, (4, 64, 768), (49152, 768, 1), 0); del buf282  # reuse
        buf322 = empty_strided_cuda((4, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_72, x_73, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_14.run(buf312, buf308, primals_76, buf314, buf322, 256, 768, stream=stream0)
        del buf308
        buf313 = empty_strided_cuda((768, 65), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_15.run(primals_2, buf313, 49920, stream=stream0)
        del primals_2
        buf315 = empty_strided_cuda((256, 65), (65, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (256, 768), (768, 1), 0), buf313, out=buf315)
        buf316 = empty_strided_cuda((256, 1), (1, 1), torch.float32)
        buf317 = empty_strided_cuda((256, 1), (1, 256), torch.float32)
        buf318 = reinterpret_tensor(buf317, (256, 1), (1, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [loss, ], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_prepare_softmax_online_16.run(buf318, buf315, buf316, 256, 65, stream=stream0)
        buf321 = empty_strided_cuda((), (), torch.float32)
        buf320 = empty_strided_cuda((), (), torch.float32)
        buf346 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_per_fused_nll_loss_forward_17.run(buf346, primals_77, buf315, buf316, buf318, buf320, 1, 256, stream=stream0)
    return (reinterpret_tensor(buf315, (4, 64, 65), (4160, 65, 1), 0), buf346, primals_1, primals_4, primals_7, primals_10, primals_13, primals_16, primals_19, primals_22, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_77, buf0, buf1, buf2, buf3, buf6, reinterpret_tensor(buf8, (256, 768), (768, 1), 0), reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf9, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf11, buf12, buf13, buf14, buf21, reinterpret_tensor(buf23, (256, 768), (768, 1), 0), buf24, reinterpret_tensor(buf26, (256, 3072), (3072, 1), 0), buf31, reinterpret_tensor(buf33, (256, 768), (768, 1), 0), reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf34, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf36, buf37, buf38, buf39, buf47, reinterpret_tensor(buf49, (256, 768), (768, 1), 0), buf50, reinterpret_tensor(buf52, (256, 3072), (3072, 1), 0), buf57, reinterpret_tensor(buf59, (256, 768), (768, 1), 0), reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf60, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf62, buf63, buf64, buf65, buf72, reinterpret_tensor(buf74, (256, 768), (768, 1), 0), buf75, reinterpret_tensor(buf77, (256, 3072), (3072, 1), 0), buf82, reinterpret_tensor(buf84, (256, 768), (768, 1), 0), reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf85, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf87, buf88, buf89, buf90, buf98, reinterpret_tensor(buf100, (256, 768), (768, 1), 0), buf101, reinterpret_tensor(buf103, (256, 3072), (3072, 1), 0), buf108, reinterpret_tensor(buf110, (256, 768), (768, 1), 0), reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf111, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf113, buf114, buf115, buf116, buf123, reinterpret_tensor(buf125, (256, 768), (768, 1), 0), buf126, reinterpret_tensor(buf128, (256, 3072), (3072, 1), 0), buf133, reinterpret_tensor(buf135, (256, 768), (768, 1), 0), reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf136, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf138, buf139, buf140, buf141, buf149, reinterpret_tensor(buf151, (256, 768), (768, 1), 0), buf152, reinterpret_tensor(buf154, (256, 3072), (3072, 1), 0), buf159, reinterpret_tensor(buf161, (256, 768), (768, 1), 0), reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf162, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf164, buf165, buf166, buf167, buf174, reinterpret_tensor(buf176, (256, 768), (768, 1), 0), buf177, reinterpret_tensor(buf179, (256, 3072), (3072, 1), 0), buf184, reinterpret_tensor(buf186, (256, 768), (768, 1), 0), reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf187, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf189, buf190, buf191, buf192, buf200, reinterpret_tensor(buf202, (256, 768), (768, 1), 0), buf203, reinterpret_tensor(buf205, (256, 3072), (3072, 1), 0), buf210, reinterpret_tensor(buf212, (256, 768), (768, 1), 0), reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf213, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf215, buf216, buf217, buf218, buf225, reinterpret_tensor(buf227, (256, 768), (768, 1), 0), buf228, reinterpret_tensor(buf230, (256, 3072), (3072, 1), 0), buf235, reinterpret_tensor(buf237, (256, 768), (768, 1), 0), reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf238, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf240, buf241, buf242, buf243, buf251, reinterpret_tensor(buf253, (256, 768), (768, 1), 0), buf254, reinterpret_tensor(buf256, (256, 3072), (3072, 1), 0), buf261, reinterpret_tensor(buf263, (256, 768), (768, 1), 0), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf264, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf266, buf267, buf268, buf269, buf276, reinterpret_tensor(buf278, (256, 768), (768, 1), 0), buf279, reinterpret_tensor(buf281, (256, 3072), (3072, 1), 0), buf286, reinterpret_tensor(buf288, (256, 768), (768, 1), 0), reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 768), reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 0), reinterpret_tensor(buf289, (4, 12, 64, 64), (147456, 64, 2304, 1), 1536), buf291, buf292, buf293, buf294, buf302, reinterpret_tensor(buf304, (256, 768), (768, 1), 0), buf305, reinterpret_tensor(buf307, (256, 3072), (3072, 1), 0), buf312, reinterpret_tensor(buf314, (256, 768), (768, 1), 0), reinterpret_tensor(buf315, (4, 64, 65), (4160, 65, 1), 0), buf316, buf318, buf320, reinterpret_tensor(buf313, (65, 768), (768, 1), 0), buf322, reinterpret_tensor(buf306, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf303, (3072, 768), (768, 1), 0), buf323, reinterpret_tensor(buf296, (768, 768), (768, 1), 0), reinterpret_tensor(buf287, (2304, 768), (768, 1), 0), buf324, reinterpret_tensor(buf280, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf277, (3072, 768), (768, 1), 0), buf325, reinterpret_tensor(buf271, (768, 768), (768, 1), 0), reinterpret_tensor(buf262, (2304, 768), (768, 1), 0), buf326, reinterpret_tensor(buf255, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf252, (3072, 768), (768, 1), 0), buf327, reinterpret_tensor(buf245, (768, 768), (768, 1), 0), reinterpret_tensor(buf236, (2304, 768), (768, 1), 0), buf328, reinterpret_tensor(buf229, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf226, (3072, 768), (768, 1), 0), buf329, reinterpret_tensor(buf220, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (2304, 768), (768, 1), 0), buf330, reinterpret_tensor(buf204, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf201, (3072, 768), (768, 1), 0), buf331, reinterpret_tensor(buf194, (768, 768), (768, 1), 0), reinterpret_tensor(buf185, (2304, 768), (768, 1), 0), buf332, reinterpret_tensor(buf178, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf175, (3072, 768), (768, 1), 0), buf333, reinterpret_tensor(buf169, (768, 768), (768, 1), 0), reinterpret_tensor(buf160, (2304, 768), (768, 1), 0), buf334, reinterpret_tensor(buf153, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf150, (3072, 768), (768, 1), 0), buf335, reinterpret_tensor(buf143, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (2304, 768), (768, 1), 0), buf336, reinterpret_tensor(buf127, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf124, (3072, 768), (768, 1), 0), buf337, reinterpret_tensor(buf118, (768, 768), (768, 1), 0), reinterpret_tensor(buf109, (2304, 768), (768, 1), 0), buf338, reinterpret_tensor(buf102, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf99, (3072, 768), (768, 1), 0), buf339, reinterpret_tensor(buf92, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (2304, 768), (768, 1), 0), buf340, reinterpret_tensor(buf76, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf73, (3072, 768), (768, 1), 0), buf341, reinterpret_tensor(buf67, (768, 768), (768, 1), 0), reinterpret_tensor(buf58, (2304, 768), (768, 1), 0), buf342, reinterpret_tensor(buf51, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf48, (3072, 768), (768, 1), 0), buf343, reinterpret_tensor(buf41, (768, 768), (768, 1), 0), reinterpret_tensor(buf32, (2304, 768), (768, 1), 0), buf344, reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf22, (3072, 768), (768, 1), 0), buf345, reinterpret_tensor(buf16, (768, 768), (768, 1), 0), reinterpret_tensor(buf7, (2304, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((65, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
