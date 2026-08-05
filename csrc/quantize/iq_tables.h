#pragma once
#include <libgguf/formats.h>

#define NGRID_IQ1S 2048
#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

extern "C"
{
  void iq2xs_init_impl(enum ggml_type type);
  void iq2xs_free_impl(enum ggml_type type);
  void iq3xs_init_impl(int grid_size);
  void iq3xs_free_impl(int grid_size);
}

void iq_tables_init(enum ggml_type type);
void iq_tables_free();

int iq2_data_index(enum ggml_type type);
int iq3_data_index(int grid_size);
int iq2_find_best_neighbour(const uint16_t *LIBGGUF_RESTRICT neighbours, const uint64_t *LIBGGUF_RESTRICT grid, const float *LIBGGUF_RESTRICT xval, const float *LIBGGUF_RESTRICT weight, float scale, int8_t *LIBGGUF_RESTRICT L);
int iq3_find_best_neighbour(const uint16_t *LIBGGUF_RESTRICT neighbours, const uint32_t *LIBGGUF_RESTRICT grid, const float *LIBGGUF_RESTRICT xval, const float *LIBGGUF_RESTRICT weight, float scale, int8_t *LIBGGUF_RESTRICT L);
int iq1_find_best_neighbour2(const uint16_t *LIBGGUF_RESTRICT neighbours, const uint64_t *LIBGGUF_RESTRICT grid, const float *LIBGGUF_RESTRICT xval, const float *LIBGGUF_RESTRICT weight, float scale, const float *LIBGGUF_RESTRICT xg, int8_t *LIBGGUF_RESTRICT L, int ngrid);
int iq1_sort_helper(const void *left, const void *right);

struct iq2_entry_t
{
  uint64_t *grid;
  int *map;
  uint16_t *neighbours;
};

struct iq3_entry_t
{
  uint32_t *grid;
  int *map;
  uint16_t *neighbours;
};

extern iq2_entry_t iq2_data[4];
extern iq3_entry_t iq3_data[2];
