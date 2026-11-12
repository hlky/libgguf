#include "conversion/converter.h"

#include "conversion/common.h"
#include "conversion/cpu_quantizer.h"
#include "cli/quantize/policy.h"
#include "conversion/safetensors.h"
#include "conversion/writer.h"
#include "conversion/gpu_quantizer.h"

#include <libgguf/libgguf.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

struct timing_totals {
  double metadata_s = 0.0;
  double read_s = 0.0;
  double cpu_convert_s = 0.0;
  double write_s = 0.0;
  double total_s = 0.0;
  uint64_t tensors = 0;
  uint64_t gpu_tensors = 0;
  uint64_t gpu_verified_tensors = 0;
  gpu_quantizer_timings gpu;
};

double elapsed_seconds(steady_clock::time_point begin, steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

std::string default_output_path(const std::string & src, const std::string & file_type_name) {
    size_t slash = src.find_last_of("/\\");
    size_t dot = src.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        dot = src.size();
    }
    return src.substr(0, dot) + "-" + file_type_name + ".gguf";
}
template <typename Fn>
void measure_phase(timing_totals *timings, double timing_totals::*field, Fn &&fn) {
  if (!timings) {
    fn();
    return;
  }
  const auto begin = steady_clock::now();
  fn();
  const auto end = steady_clock::now();
  timings->*field += elapsed_seconds(begin, end);
}

bool converter_auto_prefers_gpu_qtype(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    return true;
  default:
    return false;
  }
}

bool conversion_plans_prefer_gpu(const std::vector<tensor_plan> &plans) {
  for (const tensor_plan &plan : plans) {
    if (is_quantized_qtype(plan.qtype) && converter_auto_prefers_gpu_qtype(plan.qtype)) {
      return true;
    }
  }
  return false;
}

bool conversion_plan_requests_gpu(const tensor_plan &plan, const cli_options &options) {
  if (!is_quantized_qtype(plan.qtype)) {
    return false;
  }
  return options.backend == CONVERTER_BACKEND_GPU ||
         (options.backend == CONVERTER_BACKEND_AUTO && converter_auto_prefers_gpu_qtype(plan.qtype));
}

bool conversion_plan_uses_gpu(const tensor_plan &plan, const cli_options &options, gpu_quantizer *gpu_ctx) {
  if (!conversion_plan_requests_gpu(plan, options)) {
    return false;
  }
#if defined(LIBGGUF_HAS_GPU)
  return gpu_ctx && gpu_ctx->supports(plan.qtype, plan.n_per_row);
#else
  (void)gpu_ctx;
  return false;
#endif
}

std::set<std::string> select_gpu_verification_tensor_keys(
    const std::vector<tensor_plan> &plans,
    const cli_options &options,
    gpu_quantizer *gpu_ctx) {
  struct largest_candidate {
    uint64_t expected_nbytes = 0;
    std::string key;
    size_t index = 0;
  };
  std::set<std::string> selected;
  std::vector<largest_candidate> largest_candidates;
  std::vector<std::string> random_candidates;
  for (size_t i = 0; i < plans.size(); ++i) {
    if (conversion_plan_uses_gpu(plans[i], options, gpu_ctx)) {
      largest_candidates.push_back({plans[i].expected_nbytes, plans[i].key, i});
      random_candidates.push_back(plans[i].key);
    }
  }
  std::sort(largest_candidates.begin(), largest_candidates.end(), [](const largest_candidate &a, const largest_candidate &b) {
    if (a.expected_nbytes != b.expected_nbytes) {
      return a.expected_nbytes > b.expected_nbytes;
    }
    if (a.key != b.key) {
      return a.key < b.key;
    }
    return a.index < b.index;
  });
  const size_t selected_count =
      (size_t)std::min<uint64_t>(options.verify_gpu_large_tensors, (uint64_t)largest_candidates.size());
  for (size_t i = 0; i < selected_count; ++i) {
    selected.insert(largest_candidates[i].key);
  }
  if (options.verify_gpu_random_tensors > 0 && !random_candidates.empty()) {
    std::mt19937_64 rng(options.seed);
    std::shuffle(random_candidates.begin(), random_candidates.end(), rng);
    const size_t random_selected_count =
        (size_t)std::min<uint64_t>(options.verify_gpu_random_tensors, (uint64_t)random_candidates.size());
    for (size_t i = 0; i < random_selected_count; ++i) {
      selected.insert(random_candidates[i]);
    }
  }
  return selected;
}

void write_tensor_payload(
    FILE *out,
    const input_file &input,
    const tensor_plan &plan,
    const cli_options &options,
    cpu_quantizer &native_pool,
    gpu_quantizer *gpu_ctx,
    uint64_t *verify_gpu_remaining,
    const std::set<std::string> &verify_gpu_selected_keys,
    timing_totals *timings) {
  if (timings) {
    timings->tensors += 1;
  }
  const uint64_t scratch_bytes = options.scratch_bytes;
  write_pre_tensor_padding(out, GGUF_DEFAULT_ALIGNMENT);
  if (qtype_matches_source(plan.source_dtype, plan.qtype)) {
    const uint64_t buffer_bytes = std::max<uint64_t>(1, std::min<uint64_t>(scratch_bytes, plan.expected_nbytes));
    std::vector<uint8_t> buffer((size_t)buffer_bytes);
    uint64_t remaining = plan.expected_nbytes;
    uint64_t offset = plan.data_begin;
    while (remaining > 0) {
      const uint64_t chunk = std::min<uint64_t>(remaining, buffer.size());
      std::string error;
      bool read_ok = false;
      measure_phase(timings, &timing_totals::read_s, [&]() {
        read_ok = input.read_at(offset, buffer.data(), chunk, &error);
      });
      if (!read_ok) {
        fail(error + " for " + plan.key);
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, buffer.data(), chunk);
      });
      offset += chunk;
      remaining -= chunk;
    }
    write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
    return;
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  if (row_size == 0) {
    fail("unsupported qtype or row width for " + plan.key);
  }
  const bool source_f32 = plan.source_dtype == NATIVE_DTYPE_F32;
  const bool native_quant = is_quantized_qtype(plan.qtype) && cpu_quantizer_supports(plan.qtype);
  const bool gpu_quant = conversion_plan_uses_gpu(plan, options, gpu_ctx);
  if (conversion_plan_requests_gpu(plan, options)) {
    if (options.backend == CONVERTER_BACKEND_GPU && !gpu_quant && !options.gpu_fallback_cpu) {
      fail(
          "GPU backend does not support " + qtype_name(plan.qtype) +
          " for tensor " + plan.key + "; pass --gpu-fallback cpu to use CPU for unsupported tensors");
    }
  }
  const bool verify_gpu_first =
      gpu_quant && verify_gpu_remaining && *verify_gpu_remaining > 0;
  const bool verify_gpu_selected =
      gpu_quant && verify_gpu_selected_keys.find(plan.key) != verify_gpu_selected_keys.end();
  const bool verify_gpu = verify_gpu_first || verify_gpu_selected;
  if (gpu_quant && timings) {
    timings->gpu_tensors += 1;
    if (verify_gpu) {
      timings->gpu_verified_tensors += 1;
    }
  }
  const uint64_t row_work_bytes =
      plan.n_per_row * source_dtype_size(plan.source_dtype) +
      (source_f32 ? 0 : plan.n_per_row * sizeof(float)) +
      (uint64_t)row_size;
  uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
  if (rows_per_chunk == 0) {
    rows_per_chunk = 1;
  }
  if (gpu_quant && options.gpu_vram_bytes > 0) {
    const uint64_t device_row_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    rows_per_chunk = device_row_bytes == 0 ? 1 : options.gpu_vram_bytes / device_row_bytes;
    if (rows_per_chunk == 0) {
      rows_per_chunk = 1;
    }
  }
  rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

#if defined(LIBGGUF_HAS_GPU)
  if (gpu_quant && !verify_gpu && options.gpu_pipeline) {
    struct gpu_pipeline_slot_state {
      uint64_t row = 0;
      uint64_t row_count = 0;
      size_t expected = 0;
      std::vector<uint8_t> source_bytes;
    };

    const size_t max_input_bytes = (size_t)(rows_per_chunk * plan.n_per_row * sizeof(float));
    const size_t max_output_bytes = (size_t)(rows_per_chunk * (uint64_t)row_size);
    gpu_ctx->resize_device_buffers(max_input_bytes, max_output_bytes);
    gpu_ctx->resize_host_slot(0, max_input_bytes, max_output_bytes);
    gpu_ctx->resize_host_slot(1, max_input_bytes, max_output_bytes);

    gpu_pipeline_slot_state slots[2];
    auto prepare_and_enqueue = [&](size_t slot_index, uint64_t row) {
      gpu_pipeline_slot_state &slot = slots[slot_index];
      slot.row = row;
      slot.row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      slot.expected = (size_t)(slot.row_count * (uint64_t)row_size);
      float *host_f32 = gpu_ctx->host_input_data(slot_index);
      const uint64_t value_begin = row * plan.n_per_row;
      const uint64_t value_count = slot.row_count * plan.n_per_row;
      if (source_f32) {
        const uint64_t byte_count = value_count * sizeof(float);
        std::string error;
        bool read_ok = false;
        measure_phase(timings, &timing_totals::read_s, [&]() {
          read_ok = input.read_at(plan.data_begin + value_begin * sizeof(float), host_f32, byte_count, &error);
        });
        if (!read_ok) {
          fail(error + " for " + plan.key);
        }
      } else {
        measure_phase(timings, &timing_totals::read_s, [&]() {
          read_source_chunk(input, plan, row, slot.row_count, slot.source_bytes);
        });
        measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
          decode_source_bytes_to_f32(plan, slot.source_bytes.data(), value_count, host_f32);
        });
      }
      gpu_ctx->enqueue_quantize_chunk(
          slot_index,
          plan.qtype,
          plan.n_per_row,
          host_f32,
          slot.row_count,
          gpu_ctx->host_output_data(slot_index),
          slot.expected,
          timings ? &timings->gpu : nullptr);
    };
    auto finish_and_write = [&](size_t slot_index) {
      gpu_pipeline_slot_state &slot = slots[slot_index];
      gpu_ctx->finish_quantize_chunk(slot_index, timings ? &timings->gpu : nullptr);
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, gpu_ctx->host_output_data(slot_index), slot.expected);
      });
    };

    size_t active_slot = 0;
    uint64_t row = 0;
    prepare_and_enqueue(active_slot, row);
    row += slots[active_slot].row_count;
    while (row < plan.n_rows) {
      const size_t next_slot = 1 - active_slot;
      prepare_and_enqueue(next_slot, row);
      row += slots[next_slot].row_count;
      finish_and_write(active_slot);
      active_slot = next_slot;
    }
    finish_and_write(active_slot);
    write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
    return;
  }
#endif

  std::vector<float> scratch;
  if (!source_f32) {
    scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
  }
  std::vector<uint8_t> source_bytes;
  std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
  std::vector<uint8_t> verify_encoded;
  if (verify_gpu) {
    verify_encoded.resize(encoded.size());
  }
  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk) {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    measure_phase(timings, &timing_totals::read_s, [&]() {
      read_source_chunk(input, plan, row, row_count, source_bytes);
    });
    const float *chunk_src = nullptr;
    if (gpu_quant) {
#if defined(LIBGGUF_HAS_GPU)
      measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
        if (source_f32) {
          chunk_src = reinterpret_cast<const float *>(source_bytes.data());
        } else {
          decode_source_bytes_to_f32(plan, source_bytes.data(), row_count * plan.n_per_row, scratch.data());
          chunk_src = scratch.data();
        }
      });
      const size_t expected = (size_t)(row_count * (uint64_t)row_size);
      gpu_ctx->quantize_chunk(
          plan.qtype,
          plan.n_per_row,
          chunk_src,
          row_count,
          encoded.data(),
          expected,
          timings ? &timings->gpu : nullptr);
      if (verify_gpu) {
        const bool quant_ok = native_pool.run(plan, source_bytes.data(), row_count, verify_encoded.data(), nullptr, scratch_bytes);
        if (!quant_ok || std::memcmp(encoded.data(), verify_encoded.data(), expected) != 0) {
          fail("GPU verification mismatch for " + plan.key + " at row " + std::to_string(row));
        }
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, encoded.data(), expected);
      });
      continue;
#endif
    }
    if (native_quant) {
      bool quant_ok = false;
      measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
        quant_ok = native_pool.run(plan, source_bytes.data(), row_count, encoded.data(), nullptr, scratch_bytes);
      });
      if (!quant_ok) {
        fail("native quantization returned an unexpected byte count for " + plan.key);
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, encoded.data(), row_count * (uint64_t)row_size);
      });
      continue;
    }
    size_t written = 0;
    const uint64_t expected = row_count * (uint64_t)row_size;
    measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
      if (source_f32) {
        chunk_src = reinterpret_cast<const float *>(source_bytes.data());
      } else {
        decode_source_bytes_to_f32(plan, source_bytes.data(), row_count * plan.n_per_row, scratch.data());
        chunk_src = scratch.data();
      }
      written = libgguf_quantize_chunk(plan.qtype, chunk_src, encoded.data(), 0, (int64_t)row_count, (int64_t)plan.n_per_row, nullptr);
    });
    if (written != expected) {
      fail("libgguf_quantize_chunk returned an unexpected byte count for " + plan.key);
    }
    measure_phase(timings, &timing_totals::write_s, [&]() {
      write_all_file(out, encoded.data(), expected);
    });
  }
  if (verify_gpu_first && verify_gpu_remaining && *verify_gpu_remaining != VERIFY_GPU_ALL_TENSORS) {
    *verify_gpu_remaining -= 1;
  }
  write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
}

} // namespace

conversion_result convert(const cli_options &options) {
#if !defined(LIBGGUF_HAS_GPU)
  if (options.backend == CONVERTER_BACKEND_GPU) {
    fail("GPU backend requested, but this executable was built without native GPU support");
  }
#endif
  const auto total_begin = steady_clock::now();
  auto [file_type_name, base_qtype_name] = parse_qtype(options.qtype);
  std::string dst = options.dst.empty() ? default_output_path(options.src, file_type_name) : options.dst;
  if (file_exists(dst) && !options.overwrite) {
    fail("Output exists and overwriting is disabled: " + dst);
  }

  input_file input;
  std::string input_error;
  if (!input.open(options.src, &input_error)) {
    fail(input_error);
  }
  uint64_t data_start = 0;
  std::vector<tensor_meta> tensors = parse_safetensors_header(input, &data_start);
  (void)data_start;
  strip_prefix(tensors);
  model_template model = detect_arch(tensors);

  conversion_result result;
  result.output_path = dst;
  result.arch = model.arch;
  result.file_type_name = "MOSTLY_" + file_type_name;

  std::vector<kv_value> kv;
  kv.push_back({"general.architecture", GGUF_TYPE_STRING, 0, model.arch, {}});
  kv.push_back({"general.quantization_version", GGUF_TYPE_UINT32, GGML_QUANT_VERSION, {}, {}});
  kv.push_back({"general.file_type", GGUF_TYPE_UINT32, file_type_value(file_type_name), {}, {}});

  std::vector<tensor_info> infos;
  std::vector<tensor_plan> plans = prepare_plans(tensors, model, file_type_name, base_qtype_name, options, kv, infos, result);
  timing_totals timings;
  timing_totals *timings_ptr = options.timings ? &timings : nullptr;
  const unsigned int worker_threads = native_thread_limit(options.threads);
  cpu_quantizer native_pool(worker_threads);
#if defined(LIBGGUF_HAS_GPU)
  std::unique_ptr<gpu_quantizer> gpu_ctx;
  if (options.backend == CONVERTER_BACKEND_GPU) {
    gpu_ctx.reset(new gpu_quantizer());
  } else if (options.backend == CONVERTER_BACKEND_AUTO && conversion_plans_prefer_gpu(plans)) {
    try {
      gpu_ctx.reset(new gpu_quantizer());
    } catch (const std::runtime_error &) {
      gpu_ctx.reset();
    }
  }
#else
  gpu_quantizer *gpu_ctx = nullptr;
#endif
  const std::set<std::string> verify_gpu_selected_keys = select_gpu_verification_tensor_keys(
      plans,
      options,
#if defined(LIBGGUF_HAS_GPU)
      gpu_ctx.get()
#else
      gpu_ctx
#endif
  );
  uint64_t verify_gpu_remaining = options.verify_gpu_tensors;

#if defined(_WIN32)
  FILE *out = nullptr;
  fopen_s(&out, dst.c_str(), "wb");
#else
  FILE *out = std::fopen(dst.c_str(), "wb");
#endif
  if (!out) {
    fail("failed to open output file: " + dst);
  }
  try {
    const auto metadata_begin = steady_clock::now();
    write_gguf_metadata(out, kv, infos);
    const auto metadata_end = steady_clock::now();
    if (timings_ptr) {
      timings.metadata_s += elapsed_seconds(metadata_begin, metadata_end);
    }
    for (const tensor_plan &plan : plans) {
      write_tensor_payload(
          out,
          input,
          plan,
          options,
          native_pool,
#if defined(LIBGGUF_HAS_GPU)
          gpu_ctx.get(),
#else
          gpu_ctx,
#endif
          &verify_gpu_remaining,
          verify_gpu_selected_keys,
          timings_ptr);
    }
    if (std::fclose(out) != 0) {
      out = nullptr;
      fail("failed to close output file");
    }
    out = nullptr;
  } catch (...) {
    if (out) {
      std::fclose(out);
    }
    throw;
  }
  if (timings_ptr) {
    timings.total_s = elapsed_seconds(total_begin, steady_clock::now());
    std::fprintf(
        stderr,
        "Timings: total=%.3fs metadata=%.3fs read=%.3fs cpu_convert=%.3fs h2d=%.3fs gpu_quant=%.3fs d2h=%.3fs write=%.3fs tensors=%llu gpu_tensors=%llu gpu_verified=%llu gpu_chunks=%llu threads=%u scratch=%llu gpu_vram=%llu gpu_pipeline=%u gpu_max_input=%llu gpu_max_output=%llu\n",
        timings.total_s,
        timings.metadata_s,
        timings.read_s,
        timings.cpu_convert_s,
        timings.gpu.h2d_s,
        timings.gpu.quantize_s,
        timings.gpu.d2h_s,
        timings.write_s,
        (unsigned long long)timings.tensors,
        (unsigned long long)timings.gpu_tensors,
        (unsigned long long)timings.gpu_verified_tensors,
        (unsigned long long)timings.gpu.chunks,
        worker_threads,
        (unsigned long long)options.scratch_bytes,
        (unsigned long long)options.gpu_vram_bytes,
        options.gpu_pipeline ? 1U : 0U,
        (unsigned long long)timings.gpu.max_input_bytes,
        (unsigned long long)timings.gpu.max_output_bytes);
  }
  return result;
}
