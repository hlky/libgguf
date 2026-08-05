#include <cstdio>

#include <libgguf/libgguf.h>
#include <libgguf/formats.h>

static size_t libgguf_block_row_size(int64_t n_per_row, int64_t block_size, size_t type_size)
{
  if (n_per_row <= 0 || block_size <= 0 || n_per_row % block_size != 0)
  {
    return 0;
  }
  return (size_t)(n_per_row / block_size) * type_size;
}

extern "C" LIBGGUF_API size_t libgguf_row_size(enum ggml_type type, int64_t n_per_row)
{
  if (n_per_row <= 0)
  {
    return 0;
  }

  switch (type)
  {
  case GGML_TYPE_F32:
    return (size_t)n_per_row * sizeof(float);
  case GGML_TYPE_F16:
    return (size_t)n_per_row * sizeof(ggml_fp16_t);
  case GGML_TYPE_BF16:
    return (size_t)n_per_row * sizeof(ggml_bf16_t);
  case GGML_TYPE_Q1_0:
    return libgguf_block_row_size(n_per_row, QK1_0, sizeof(block_q1_0));
  case GGML_TYPE_Q4_0:
    return libgguf_block_row_size(n_per_row, QK4_0, sizeof(block_q4_0));
  case GGML_TYPE_Q4_1:
    return libgguf_block_row_size(n_per_row, QK4_1, sizeof(block_q4_1));
  case GGML_TYPE_Q5_0:
    return libgguf_block_row_size(n_per_row, QK5_0, sizeof(block_q5_0));
  case GGML_TYPE_Q5_1:
    return libgguf_block_row_size(n_per_row, QK5_1, sizeof(block_q5_1));
  case GGML_TYPE_Q8_0:
    return libgguf_block_row_size(n_per_row, QK8_0, sizeof(block_q8_0));
  case GGML_TYPE_Q2_K:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_q2_K));
  case GGML_TYPE_Q3_K:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_q3_K));
  case GGML_TYPE_Q4_K:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_q4_K));
  case GGML_TYPE_Q5_K:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_q5_K));
  case GGML_TYPE_Q6_K:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_q6_K));
  case GGML_TYPE_IQ2_XXS:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq2_xxs));
  case GGML_TYPE_IQ2_XS:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq2_xs));
  case GGML_TYPE_IQ2_S:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq2_s));
  case GGML_TYPE_IQ3_XXS:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq3_xxs));
  case GGML_TYPE_IQ3_S:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq3_s));
  case GGML_TYPE_IQ1_S:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq1_s));
  case GGML_TYPE_IQ1_M:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq1_m));
  case GGML_TYPE_IQ4_NL:
    return libgguf_block_row_size(n_per_row, QK4_NL, sizeof(block_iq4_nl));
  case GGML_TYPE_IQ4_XS:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_iq4_xs));
  case GGML_TYPE_TQ1_0:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_tq1_0));
  case GGML_TYPE_TQ2_0:
    return libgguf_block_row_size(n_per_row, QK_K, sizeof(block_tq2_0));
  case GGML_TYPE_MXFP4:
    return libgguf_block_row_size(n_per_row, QK_MXFP4, sizeof(block_mxfp4));
  case GGML_TYPE_NVFP4:
    return libgguf_block_row_size(n_per_row, QK_NVFP4, sizeof(block_nvfp4));
  default:
    return 0;
  }
}

extern "C" LIBGGUF_API int64_t libgguf_block_size(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
    return 1;
  case GGML_TYPE_Q1_0:
    return QK1_0;
  case GGML_TYPE_Q4_0:
    return QK4_0;
  case GGML_TYPE_Q4_1:
    return QK4_1;
  case GGML_TYPE_Q5_0:
    return QK5_0;
  case GGML_TYPE_Q5_1:
    return QK5_1;
  case GGML_TYPE_Q8_0:
    return QK8_0;
  case GGML_TYPE_IQ4_NL:
    return QK4_NL;
  case GGML_TYPE_MXFP4:
    return QK_MXFP4;
  case GGML_TYPE_NVFP4:
    return QK_NVFP4;
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
    return QK_K;
  default:
    return 0;
  }
}

extern "C" LIBGGUF_API bool libgguf_quantize_requires_imatrix(enum ggml_type type)
{
  return type == GGML_TYPE_IQ2_XXS ||
         type == GGML_TYPE_IQ2_XS ||
         type == GGML_TYPE_IQ1_S;
}

extern "C" LIBGGUF_API size_t libgguf_type_size(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_Q1_0:
    return sizeof(block_q1_0);
  case GGML_TYPE_Q4_0:
    return sizeof(block_q4_0);
  case GGML_TYPE_Q4_1:
    return sizeof(block_q4_1);
  case GGML_TYPE_Q5_0:
    return sizeof(block_q5_0);
  case GGML_TYPE_Q5_1:
    return sizeof(block_q5_1);
  case GGML_TYPE_Q8_0:
    return sizeof(block_q8_0);
  case GGML_TYPE_Q8_1:
    return sizeof(block_q8_1);
  case GGML_TYPE_Q2_K:
    return sizeof(block_q2_K);
  case GGML_TYPE_Q3_K:
    return sizeof(block_q3_K);
  case GGML_TYPE_Q4_K:
    return sizeof(block_q4_K);
  case GGML_TYPE_Q5_K:
    return sizeof(block_q5_K);
  case GGML_TYPE_Q6_K:
    return sizeof(block_q6_K);
  case GGML_TYPE_Q8_K:
    return sizeof(block_q8_K);
  case GGML_TYPE_IQ2_XXS:
    return sizeof(block_iq2_xxs);
  case GGML_TYPE_IQ2_XS:
    return sizeof(block_iq2_xs);
  case GGML_TYPE_IQ3_XXS:
    return sizeof(block_iq3_xxs);
  case GGML_TYPE_IQ1_S:
    return sizeof(block_iq1_s);
  case GGML_TYPE_IQ4_NL:
    return sizeof(block_iq4_nl);
  case GGML_TYPE_IQ3_S:
    return sizeof(block_iq3_s);
  case GGML_TYPE_IQ2_S:
    return sizeof(block_iq2_s);
  case GGML_TYPE_IQ4_XS:
    return sizeof(block_iq4_xs);
  case GGML_TYPE_IQ1_M:
    return sizeof(block_iq1_m);
  case GGML_TYPE_TQ1_0:
    return sizeof(block_tq1_0);
  case GGML_TYPE_TQ2_0:
    return sizeof(block_tq2_0);
  case GGML_TYPE_MXFP4:
    return sizeof(block_mxfp4);
  case GGML_TYPE_NVFP4:
    return sizeof(block_nvfp4);
  case GGML_TYPE_F16:
    return sizeof(ggml_fp16_t);
  case GGML_TYPE_BF16:
    return sizeof(ggml_bf16_t);
  case GGML_TYPE_F32:
    return sizeof(float);
  case GGML_TYPE_F64:
    return sizeof(double);
  case GGML_TYPE_I8:
    return sizeof(int8_t);
  case GGML_TYPE_I16:
    return sizeof(int16_t);
  case GGML_TYPE_I32:
    return sizeof(int32_t);
  case GGML_TYPE_I64:
    return sizeof(int64_t);
  default:
    return 0;
  }
}

extern "C" LIBGGUF_API const char *libgguf_type_name(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_F32:
    return "f32";
  case GGML_TYPE_F16:
    return "f16";
  case GGML_TYPE_Q4_0:
    return "q4_0";
  case GGML_TYPE_Q4_1:
    return "q4_1";
  case GGML_TYPE_Q5_0:
    return "q5_0";
  case GGML_TYPE_Q5_1:
    return "q5_1";
  case GGML_TYPE_Q8_0:
    return "q8_0";
  case GGML_TYPE_Q8_1:
    return "q8_1";
  case GGML_TYPE_Q2_K:
    return "q2_K";
  case GGML_TYPE_Q3_K:
    return "q3_K";
  case GGML_TYPE_Q4_K:
    return "q4_K";
  case GGML_TYPE_Q5_K:
    return "q5_K";
  case GGML_TYPE_Q6_K:
    return "q6_K";
  case GGML_TYPE_Q8_K:
    return "q8_K";
  case GGML_TYPE_IQ2_XXS:
    return "iq2_xxs";
  case GGML_TYPE_IQ2_XS:
    return "iq2_xs";
  case GGML_TYPE_IQ3_XXS:
    return "iq3_xxs";
  case GGML_TYPE_IQ1_S:
    return "iq1_s";
  case GGML_TYPE_IQ4_NL:
    return "iq4_nl";
  case GGML_TYPE_IQ3_S:
    return "iq3_s";
  case GGML_TYPE_IQ2_S:
    return "iq2_s";
  case GGML_TYPE_IQ4_XS:
    return "iq4_xs";
  case GGML_TYPE_I8:
    return "i8";
  case GGML_TYPE_I16:
    return "i16";
  case GGML_TYPE_I32:
    return "i32";
  case GGML_TYPE_I64:
    return "i64";
  case GGML_TYPE_F64:
    return "f64";
  case GGML_TYPE_IQ1_M:
    return "iq1_m";
  case GGML_TYPE_BF16:
    return "bf16";
  case GGML_TYPE_TQ1_0:
    return "tq1_0";
  case GGML_TYPE_TQ2_0:
    return "tq2_0";
  case GGML_TYPE_MXFP4:
    return "mxfp4";
  case GGML_TYPE_NVFP4:
    return "nvfp4";
  case GGML_TYPE_Q1_0:
    return "q1_0";
  default:
    return "unknown";
  }
}
