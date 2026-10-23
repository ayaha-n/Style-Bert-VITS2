"""resize_antialias_impl.cu の _Setup*UpsampleFilterAntiAlias カーネルの std::tuple 引数を cuda::std::tuple にする。
CUDA 11.4 の nvcc は、std::tuple を値渡しするこれらのカーネルで内部エラー
(Call parameter type does not match function signature!) を起こすため。計算の中身は変えない。"""
import re, sys
path = sys.argv[1]
s = open(path).read()
if "ToCudaTuple" in s:
    print("already patched"); sys.exit(0)

lines = s.split("\n")
def find(pat, start=0):
    for i in range(start, len(lines)):
        if pat in lines[i]: return i
    raise SystemExit("not found: " + pat)

# 1. 2 つのカーネルの定義 (シグネチャから本体の終わりまで) で std::tuple / std::get を cuda::std に
for name in ("__global__ void _SetupBilinearUpsampleFilterAntiAlias(", "__global__ void _SetupTrilinerarUpsampleFilterAntiAlias("):
    b = find(name)
    e = b
    while lines[e] != "}": e += 1
    for i in range(b, e + 1):
        lines[i] = lines[i].replace("std::tuple<", "::cuda::std::tuple<").replace("std::get<", "::cuda::std::get<")

# 2. カーネルの起動箇所: <<<...>>>( から ); まで、引数を ToCudaTuple で包む / make_tuple を cuda::std に
n_launch = 0
i = 0
while i < len(lines):
    if re.search(r"_Setup(Bilinear|Trilinerar)UpsampleFilterAntiAlias<", lines[i]) and "__global__" not in lines[i]:
        j = i
        while "<<<" not in lines[j]: j += 1
        k = j + 1
        while not lines[k].rstrip().endswith("));"): k += 1
        for t in range(j + 1, k + 1):
            l = lines[t].replace("std::make_tuple(", "::cuda::std::make_tuple(")
            l = re.sub(r"^(\s*)(inferred_(input_dims|output_dims|dim_rscales)),", r"\1ToCudaTuple(\2),", l)
            lines[t] = l
        n_launch += 1
        i = k
    i += 1

s = "\n".join(lines)
# 3. ヘッダとヘルパー
helper = '''#include <cuda/std/tuple>

namespace onnxruntime {
namespace cuda {
// std::tuple を値渡しするカーネルは CUDA 11.4 の nvcc で内部エラーになるので、カーネル引数には cuda::std::tuple を使う
template <typename... Ts>
static ::cuda::std::tuple<Ts...> ToCudaTuple(const std::tuple<Ts...>& t) {
  return std::apply([](auto... v) { return ::cuda::std::make_tuple(v...); }, t);
}
}  // namespace cuda
}  // namespace onnxruntime
'''
anchor = '#include "core/providers/cuda/tensor/resize_impl.h"\n'
assert anchor in s
s = s.replace(anchor, anchor + helper, 1)
open(path, "w").write(s)
print("patched launches:", n_launch)
