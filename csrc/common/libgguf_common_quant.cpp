#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstdint>
#include <cstring>

struct libgguf_common_quant_backend_fns
{
  const char *name;
  libgguf_fp16_to_fp32_fn fp16_to_fp32;
  libgguf_fp32_to_fp16_fn fp32_to_fp16;
  libgguf_bf16_to_fp32_fn bf16_to_fp32;
  libgguf_fp32_to_bf16_fn fp32_to_bf16;
  libgguf_e8m0_to_fp32_fn e8m0_to_fp32;
  libgguf_e8m0_to_fp32_fn e8m0_to_fp32_half;
  libgguf_ue4m3_to_fp32_fn ue4m3_to_fp32;
  libgguf_fp32_to_ue4m3_fn fp32_to_ue4m3;
  libgguf_best_index_int8_fn best_index_int8;
  libgguf_best_index_mxfp4_fn best_index_mxfp4;
  libgguf_nearest_int_fn nearest_int;
  libgguf_get_scale_min_k4_fn get_scale_min_k4;
  libgguf_make_qx_quants_fn make_qx;
  libgguf_make_q3_quants_fn make_q3;
  libgguf_make_qkx_quants_fn make_qkx2;
  libgguf_make_qkx_quants_fn make_qkx3;
  libgguf_make_qp_quants_fn make_qp;
};

#define LIBGGUF_COMMON_SMALL_BACKEND_DECLS(backend)                                                                 \
  extern "C" float libgguf_fp16_to_fp32_##backend(ggml_fp16_t h);                                                   \
  extern "C" ggml_fp16_t libgguf_fp32_to_fp16_##backend(float f);                                                   \
  extern "C" float libgguf_bf16_to_fp32_##backend(ggml_bf16_t h);                                                   \
  extern "C" ggml_bf16_t libgguf_fp32_to_bf16_##backend(float f);                                                   \
  extern "C" float libgguf_e8m0_to_fp32_##backend(uint8_t x);                                                       \
  extern "C" float libgguf_e8m0_to_fp32_half_##backend(uint8_t x);                                                  \
  extern "C" float libgguf_ue4m3_to_fp32_##backend(uint8_t x);                                                      \
  extern "C" uint8_t libgguf_fp32_to_ue4m3_##backend(float x);                                                      \
  extern "C" int libgguf_best_index_int8_##backend(int n, const int8_t *val, float x);                              \
  extern "C" int libgguf_best_index_mxfp4_##backend(float x, float e);                                              \
  extern "C" int libgguf_nearest_int_##backend(float fval);                                                         \
  extern "C" void libgguf_get_scale_min_k4_##backend(int j, const uint8_t *RESTRICT q, uint8_t *RESTRICT d,         \
                                                       uint8_t *RESTRICT m);

LIBGGUF_COMMON_SMALL_BACKEND_DECLS(sse2)
LIBGGUF_COMMON_SMALL_BACKEND_DECLS(sse4_1)
LIBGGUF_COMMON_SMALL_BACKEND_DECLS(avx2)

extern "C" float libgguf_make_qx_quants_sse2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              int rmse_type, const float *RESTRICT qw);
extern "C" float libgguf_make_q3_quants_sse2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              bool do_rmse);
extern "C" float libgguf_make_qkx2_quants_sse2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qkx3_quants_sse2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qp_quants_sse2(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                              const float *quant_weights);

extern "C" float libgguf_make_qx_quants_sse4_1(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                int rmse_type, const float *RESTRICT qw);
extern "C" float libgguf_make_q3_quants_sse4_1(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                bool do_rmse);
extern "C" float libgguf_make_qkx2_quants_sse4_1(int n, int nmax, const float *RESTRICT x,
                                                  const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                  float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                  float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qkx3_quants_sse4_1(int n, int nmax, const float *RESTRICT x,
                                                  const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                  float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                  float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qp_quants_sse4_1(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                                const float *quant_weights);

extern "C" float libgguf_make_qx_quants_avx2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              int rmse_type, const float *RESTRICT qw);
extern "C" float libgguf_make_q3_quants_avx2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              bool do_rmse);
extern "C" float libgguf_make_qkx2_quants_avx2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qkx3_quants_avx2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qp_quants_avx2(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                              const float *quant_weights);

static const libgguf_common_quant_backend_fns REF_BACKEND = {
    "ref",
    ggml_compute_fp16_to_fp32,
    ggml_compute_fp32_to_fp16,
    ggml_compute_bf16_to_fp32,
    ggml_compute_fp32_to_bf16,
    ggml_e8m0_to_fp32,
    ggml_e8m0_to_fp32_half,
    ggml_ue4m3_to_fp32,
    ggml_fp32_to_ue4m3,
    best_index_int8,
    best_index_mxfp4,
    nearest_int,
    get_scale_min_k4,
    libgguf_make_qx_quants_ref,
    libgguf_make_q3_quants_ref,
    libgguf_make_qkx2_quants_ref,
    libgguf_make_qkx3_quants_ref,
    libgguf_make_qp_quants_ref,
};

static const libgguf_common_quant_backend_fns *libgguf_common_quant_backend_for_name(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
  {
    return &REF_BACKEND;
  }
  static const libgguf_common_quant_backend_fns SSE2_BACKEND = {
      "sse2",
      libgguf_fp16_to_fp32_sse2,
      libgguf_fp32_to_fp16_sse2,
      libgguf_bf16_to_fp32_sse2,
      libgguf_fp32_to_bf16_sse2,
      libgguf_e8m0_to_fp32_sse2,
      libgguf_e8m0_to_fp32_half_sse2,
      libgguf_ue4m3_to_fp32_sse2,
      libgguf_fp32_to_ue4m3_sse2,
      libgguf_best_index_int8_sse2,
      libgguf_best_index_mxfp4_sse2,
      libgguf_nearest_int_sse2,
      libgguf_get_scale_min_k4_sse2,
      libgguf_make_qx_quants_sse2,
      libgguf_make_q3_quants_sse2,
      libgguf_make_qkx2_quants_sse2,
      libgguf_make_qkx3_quants_sse2,
      libgguf_make_qp_quants_sse2,
  };
  static const libgguf_common_quant_backend_fns SSE4_1_BACKEND = {
      "sse4_1",
      libgguf_fp16_to_fp32_sse4_1,
      libgguf_fp32_to_fp16_sse4_1,
      libgguf_bf16_to_fp32_sse4_1,
      libgguf_fp32_to_bf16_sse4_1,
      libgguf_e8m0_to_fp32_sse4_1,
      libgguf_e8m0_to_fp32_half_sse4_1,
      libgguf_ue4m3_to_fp32_sse4_1,
      libgguf_fp32_to_ue4m3_sse4_1,
      libgguf_best_index_int8_sse4_1,
      libgguf_best_index_mxfp4_sse4_1,
      libgguf_nearest_int_sse4_1,
      libgguf_get_scale_min_k4_sse4_1,
      libgguf_make_qx_quants_sse4_1,
      libgguf_make_q3_quants_sse4_1,
      libgguf_make_qkx2_quants_sse4_1,
      libgguf_make_qkx3_quants_sse4_1,
      libgguf_make_qp_quants_sse4_1,
  };
  static const libgguf_common_quant_backend_fns AVX2_BACKEND = {
      "avx2",
      libgguf_fp16_to_fp32_avx2,
      libgguf_fp32_to_fp16_avx2,
      libgguf_bf16_to_fp32_avx2,
      libgguf_fp32_to_bf16_avx2,
      libgguf_e8m0_to_fp32_avx2,
      libgguf_e8m0_to_fp32_half_avx2,
      libgguf_ue4m3_to_fp32_avx2,
      libgguf_fp32_to_ue4m3_avx2,
      libgguf_best_index_int8_avx2,
      libgguf_best_index_mxfp4_avx2,
      libgguf_nearest_int_avx2,
      libgguf_get_scale_min_k4_avx2,
      libgguf_make_qx_quants_avx2,
      libgguf_make_q3_quants_avx2,
      libgguf_make_qkx2_quants_avx2,
      libgguf_make_qkx3_quants_avx2,
      libgguf_make_qp_quants_avx2,
  };

  if (std::strcmp(backend, "sse2") == 0 && features.sse2)
  {
    return &SSE2_BACKEND;
  }
  if (std::strcmp(backend, "sse4_1") == 0 && features.sse4_1)
  {
    return &SSE4_1_BACKEND;
  }
  if (std::strcmp(backend, "avx2") == 0 && features.avx2)
  {
    return &AVX2_BACKEND;
  }
  return nullptr;
}

static const libgguf_common_quant_backend_fns *&libgguf_common_quant_selected_slot()
{
  static const libgguf_common_quant_backend_fns *selected = &REF_BACKEND;
  return selected;
}

static const libgguf_common_quant_backend_fns &libgguf_common_quant_selected()
{
  return *libgguf_common_quant_selected_slot();
}

extern "C" const char *libgguf_common_quant_backend(void)
{
  return libgguf_common_quant_selected().name;
}

extern "C" int libgguf_common_quant_cpu_supports_backend(const char *backend)
{
  return libgguf_common_quant_backend_for_name(backend) ? 1 : 0;
}

extern "C" int libgguf_common_quant_set_backend(const char *backend)
{
  const libgguf_common_quant_backend_fns *selected = libgguf_common_quant_backend_for_name(backend);
  if (!selected)
  {
    return 0;
  }
  libgguf_common_quant_selected_slot() = selected;
  return 1;
}

static void libgguf_probe_mix_u64(uint64_t *hash, uint64_t value)
{
  *hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (*hash << 6) + (*hash >> 2);
}

static void libgguf_probe_mix_float(uint64_t *hash, float value)
{
  uint32_t bits = fp32_to_bits(value);
  libgguf_probe_mix_u64(hash, bits);
}

extern "C" uint64_t libgguf_common_quant_probe_for_backend(const char *backend)
{
  const libgguf_common_quant_backend_fns *fns = libgguf_common_quant_backend_for_name(backend);
  if (!fns)
  {
    return 0;
  }

  uint64_t hash = UINT64_C(1469598103934665603);
  const ggml_fp16_t fp16_values[] = {0x0000, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0x0400};
  for (ggml_fp16_t value : fp16_values)
  {
    libgguf_probe_mix_float(&hash, fns->fp16_to_fp32(value));
  }
  const float fp32_values[] = {0.0f, -0.0f, 1.0f, -2.5f, 0.33325195f, 65504.0f, 448.0f, 1.0e-8f};
  for (float value : fp32_values)
  {
    libgguf_probe_mix_u64(&hash, fns->fp32_to_fp16(value));
    libgguf_probe_mix_u64(&hash, fns->fp32_to_bf16(value).bits);
    libgguf_probe_mix_u64(&hash, fns->fp32_to_ue4m3(value));
  }
  const ggml_bf16_t bf16_values[] = {{0x0000}, {0x3f80}, {0xc020}, {0x7f7f}};
  for (ggml_bf16_t value : bf16_values)
  {
    libgguf_probe_mix_float(&hash, fns->bf16_to_fp32(value));
  }
  for (uint8_t value : {uint8_t(0), uint8_t(1), uint8_t(2), uint8_t(127), uint8_t(254)})
  {
    libgguf_probe_mix_float(&hash, fns->e8m0_to_fp32(value));
    libgguf_probe_mix_float(&hash, fns->e8m0_to_fp32_half(value));
    libgguf_probe_mix_float(&hash, fns->ue4m3_to_fp32(value & 0x7f));
  }

  const int8_t lut[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  for (float value : {-200.0f, -80.5f, -10.0f, -4.5f, 0.0f, 17.0f, 80.0f, 200.0f})
  {
    libgguf_probe_mix_u64(&hash, fns->best_index_int8(16, lut, value));
    libgguf_probe_mix_u64(&hash, fns->best_index_mxfp4(value, 0.5f));
    libgguf_probe_mix_u64(&hash, (uint32_t)fns->nearest_int(value * 0.125f));
  }

  const uint8_t scales[12] = {1, 2, 3, 4, 5, 6, 7, 8, 0x91, 0xa2, 0xb3, 0xc4};
  for (int j = 0; j < 8; ++j)
  {
    uint8_t d = 0;
    uint8_t m = 0;
    fns->get_scale_min_k4(j, scales, &d, &m);
    libgguf_probe_mix_u64(&hash, ((uint64_t)d << 8) | m);
  }
  return hash;
}

extern "C" float libgguf_fp16_to_fp32_dispatch(ggml_fp16_t h)
{
  return libgguf_common_quant_selected().fp16_to_fp32(h);
}

extern "C" ggml_fp16_t libgguf_fp32_to_fp16_dispatch(float f)
{
  return libgguf_common_quant_selected().fp32_to_fp16(f);
}

extern "C" float libgguf_bf16_to_fp32_dispatch(ggml_bf16_t h)
{
  return libgguf_common_quant_selected().bf16_to_fp32(h);
}

extern "C" ggml_bf16_t libgguf_fp32_to_bf16_dispatch(float f)
{
  return libgguf_common_quant_selected().fp32_to_bf16(f);
}

extern "C" float libgguf_e8m0_to_fp32_dispatch(uint8_t x)
{
  return libgguf_common_quant_selected().e8m0_to_fp32(x);
}

extern "C" float libgguf_e8m0_to_fp32_half_dispatch(uint8_t x)
{
  return libgguf_common_quant_selected().e8m0_to_fp32_half(x);
}

extern "C" float libgguf_ue4m3_to_fp32_dispatch(uint8_t x)
{
  return libgguf_common_quant_selected().ue4m3_to_fp32(x);
}

extern "C" uint8_t libgguf_fp32_to_ue4m3_dispatch(float x)
{
  return libgguf_common_quant_selected().fp32_to_ue4m3(x);
}

extern "C" int libgguf_best_index_int8_dispatch(int n, const int8_t *val, float x)
{
  return libgguf_common_quant_selected().best_index_int8(n, val, x);
}

extern "C" int libgguf_best_index_mxfp4_dispatch(float x, float e)
{
  return libgguf_common_quant_selected().best_index_mxfp4(x, e);
}

extern "C" int libgguf_nearest_int_dispatch(float fval)
{
  return libgguf_common_quant_selected().nearest_int(fval);
}

extern "C" void libgguf_get_scale_min_k4_dispatch(int j, const uint8_t *RESTRICT q, uint8_t *RESTRICT d,
                                                   uint8_t *RESTRICT m)
{
  libgguf_common_quant_selected().get_scale_min_k4(j, q, d, m);
}

extern "C" float libgguf_make_qx_quants_dispatch(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                  int rmse_type, const float *RESTRICT qw)
{
  return libgguf_common_quant_selected().make_qx(n, nmax, x, L, rmse_type, qw);
}

extern "C" float libgguf_make_q3_quants_dispatch(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                  bool do_rmse)
{
  return libgguf_common_quant_selected().make_q3(n, nmax, x, L, do_rmse);
}

extern "C" float libgguf_make_qkx2_quants_dispatch(int n, int nmax, const float *RESTRICT x,
                                                    const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                    float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                    float rmin, float rdelta, int nstep, bool use_mad)
{
  return libgguf_common_quant_selected().make_qkx2(n, nmax, x, weights, L, the_min, Laux, rmin, rdelta, nstep, use_mad);
}

extern "C" float libgguf_make_qkx3_quants_dispatch(int n, int nmax, const float *RESTRICT x,
                                                    const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                    float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                    float rmin, float rdelta, int nstep, bool use_mad)
{
  return libgguf_common_quant_selected().make_qkx3(n, nmax, x, weights, L, the_min, Laux, rmin, rdelta, nstep, use_mad);
}

extern "C" float libgguf_make_qp_quants_dispatch(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                                  const float *quant_weights)
{
  return libgguf_common_quant_selected().make_qp(n, nmax, x, L, quant_weights);
}
