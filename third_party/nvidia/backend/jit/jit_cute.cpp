#include <cstdio>
#include <cstdarg>
#include <cute/layout.hpp>
using namespace cute;

__host__ __device__ inline void host_printf(const char* fmt, ...) {
#ifndef __CUDA_ARCH__
  va_list ap;
  va_start(ap, fmt);
  int n = vprintf(fmt, ap);
  va_end(ap);
#else
  return; // 设备侧不会编译到这里；留返回值避免警告
#endif
}

template <class T>
__host__ __device__ inline void host_cute_print(T const& x) {
#ifndef __CUDA_ARCH__
  cute::print(x);
#else
  (void)x; // device 上不做任何事
#endif
}

__host__  __device__ inline auto __nv_cute_get_idx_base_handler() {
  constexpr int sizePerThread = sizePerThread_;
  constexpr int threadsPerWarp = threadsPerWarp_;
  constexpr int warpsPerCTA = warpsPerCTA_;
  constexpr int tensorShapeSize = tensorShapeSize_;

  auto thr_layout = make_layout(make_shape(Int<warpsPerCTA * threadsPerWarp>{}), make_stride(Int<1>{}));
  auto val_layout = make_layout(make_shape(Int<sizePerThread>{}), make_stride(Int<1>{}));
  // tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout);
  // -------- 2) 由 (thr,val) 求 TV-layout 和 tiler --------
  // cutedsl: tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
  //
  // C++ 里用 CUTE 代数来构造同义物：
  // - 先做 raked_product 得到 (M,N) 的“逻辑划分”，
  // - tiler_mn = product_each(shape(.)) 得到 tile 大小，
  // - 再用 right_inverse + composition 得到 (tid,val) -> (M,N) 的 TV-layout。
  auto layout_mn = raked_product(thr_layout, val_layout);       // (tid,val) ⟶ (M,N)
  host_printf("layout_mn: "); host_cute_print(layout_mn); host_printf("\n");
  auto tiler_mn  = product_each(shape(layout_mn));              // (TileM, TileN) 或 1D 时为 (Tile)
  // 临时 (tid,val) 的列主序布局：stride=(1, thr_size)
  auto tmp_tv = make_layout(
      make_shape(size(thr_layout), size(val_layout)),
      make_stride(Int<1>{}, size(thr_layout)));
  host_printf("tmp_tv: "); host_cute_print(tmp_tv); host_printf("\n");
  // TV-layout: (tid,val) -> (M,N)
  auto tv_layout = composition(tmp_tv, right_inverse(layout_mn));
  host_printf("inv(layout_mn): "); host_cute_print(right_inverse(layout_mn)); host_printf("\n");

  // 打印：和 cutedsl 的 `print` 对齐
  host_printf("Tiler: "); host_cute_print(tiler_mn); host_printf("\n");
  host_printf("TV Layout: "); host_cute_print(tv_layout); host_printf("\n");

  auto GMEM_1D = make_layout(make_shape(tensorShapeSize), make_stride(Int<1>{}));
  auto tiles = zipped_divide(GMEM_1D, tiler_mn);

  // auto LL = composition(layout2, layout1);
  // auto LL = layout2.compose(layout1, _);
  auto L = logical_divide(tiles, tv_layout);
  // auto LL = composition(tiles, make_tile(tv_layout, _));
  host_printf("L: "); host_cute_print(L); host_printf("\n");

  return L;
}

static inline auto& __nv_cute_get_idx_base_handler_wrapper() {
  static auto layout = []() {
      // 计算布局
      // ...
      return __nv_cute_get_idx_base_handler();
  }();
  return layout;
}

extern "C" int get_cute_layout_size(int *data) {
  auto L = __nv_cute_get_idx_base_handler_wrapper();

  int p_size = int(size(get<1>(L)));
  int v_size = int(size(get<1>(get<0>(L))));

  return p_size * v_size;
}

extern "C" void calculate_cute_layout(int *data) {
  auto L = __nv_cute_get_idx_base_handler_wrapper();

  // 演示：tid=0，打印 val=0..3, page=0..1 的 8 个偏移
  int tid = 0;
  int i = 0;
  printf("Index Offset: ");
  for (int p = 0; p < int(size(get<1>(L))); ++p) {
    for (int v = 0; v < int(size(get<1>(get<0>(L)))); ++v) {
      int off = L(make_coord( make_coord(tid, v), p ));     // C++ 里 layout 是可调用的
      printf("%d ", off);
      data[i++] = off;
    }
  }
  printf("\n");
}

extern "C" __device__ int __nv_cute_get_idx_base(int tid) {
   auto L = __nv_cute_get_idx_base_handler();
   return L(make_coord( make_coord(tid, 0), 0 ));
}
