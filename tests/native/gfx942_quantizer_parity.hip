#include <hip/hip_runtime.h>
#include <libgguf/rocm/gfx942/kernels/decode_pipeline.cuh>
#include <libgguf/libgguf.h>
#include "quantize/gpu.h"
#include "dequantize/gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t status) {
  if(status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

static void parity(ggml_type type, int blocks, int rows=1, bool weighted=true) {
  const int width=int(libgguf_block_size(type))*blocks;
  const int count=rows*width;
  const auto bytes=rows*libgguf_row_size(type,width);
  std::vector<float> input(count), importance(count);
  for(int i=0;i<count;++i) {
    const int block=i/int(libgguf_block_size(type));
    const float scale=block%4==0 ? 0.125F : block%4==1 ? 31.0F : block%4==2 ? 0.0F : 7.0F;
    input[i]=float((i*17)%31-15)*scale;
    importance[i]=1.0F+float(i%7)/16;
  }
  std::vector<uint8_t> expected(bytes), guarded(bytes+128,0xa5);
  if(libgguf_quantize_chunk(type,input.data(),expected.data(),0,rows,width,weighted ? importance.data() : nullptr)!=bytes)
    throw std::runtime_error("REF quantizer returned an unexpected byte count");
  float *device_input=nullptr,*device_importance=nullptr,*device_decoded=nullptr;
  uint8_t* device_storage=nullptr;
  check(hipMalloc(&device_input,count*sizeof(float)));
  check(hipMalloc(&device_importance,count*sizeof(float)));
  check(hipMalloc(&device_storage,guarded.size()));
  check(hipMalloc(&device_decoded,(count+32)*sizeof(float)));
  check(hipMemcpy(device_input,input.data(),count*sizeof(float),hipMemcpyHostToDevice));
  check(hipMemcpy(device_importance,importance.data(),count*sizeof(float),hipMemcpyHostToDevice));
  check(hipMemcpy(device_storage,guarded.data(),guarded.size(),hipMemcpyHostToDevice));
  quantize_row(device_input,weighted ? device_importance : nullptr,device_storage+64,type,count,width,nullptr);
  check(hipGetLastError()); check(hipDeviceSynchronize());
  check(hipMemcpy(guarded.data(),device_storage,guarded.size(),hipMemcpyDeviceToHost));
  if(!std::all_of(guarded.begin(),guarded.begin()+64,[](uint8_t b){return b==0xa5;}) ||
     !std::all_of(guarded.end()-64,guarded.end(),[](uint8_t b){return b==0xa5;}))
    throw std::runtime_error("quantizer modified an output guard");
  const auto mismatch=std::mismatch(expected.begin(),expected.end(),guarded.begin()+64);
  if(mismatch.first!=expected.end())
    throw std::runtime_error(std::string(libgguf_type_name(type)) + " GPU quantization differs from REF at byte " +
          std::to_string(mismatch.first-expected.begin()) + ", expected " + std::to_string(*mismatch.first) +
          ", got " + std::to_string(*mismatch.second) + ", blocks per row " + std::to_string(blocks) +
          ", rows " + std::to_string(rows) + ", weighted " + std::to_string(weighted));
  std::vector<float> oracle(count),decoded(count+32,-12345);
  libgguf_dequantize_chunk(type,expected.data(),oracle.data(),0,rows,width);
  check(hipMemcpy(device_decoded,decoded.data(),decoded.size()*sizeof(float),hipMemcpyHostToDevice));
  dequantize_row(device_storage+64,device_decoded+16,type,count,libgguf_gpu_dtype::f32,nullptr);
  check(hipGetLastError()); check(hipDeviceSynchronize());
  check(hipMemcpy(decoded.data(),device_decoded,decoded.size()*sizeof(float),hipMemcpyDeviceToHost));
  for(int i=0;i<16;++i) if(decoded[i]!=-12345 || decoded[count+16+i]!=-12345)
    throw std::runtime_error("decoder modified an output guard");
  for(int i=0;i<count;++i) if(!std::isfinite(decoded[16+i]) || std::abs(decoded[16+i]-oracle[i])>1e-5F*std::max(1.0F,std::abs(oracle[i])))
    throw std::runtime_error("GPU decoded value differs from REF");
  for(int schedule=0;schedule<4;++schedule) {
    std::fill(decoded.begin(),decoded.end(),-12345);
    check(hipMemcpy(device_decoded,decoded.data(),decoded.size()*sizeof(float),hipMemcpyHostToDevice));
    if(schedule==0)check((libgguf::rocm::gfx942::launch_decode<128,1>(type,device_storage+64,device_decoded+16,count,nullptr)));
    if(schedule==1)check((libgguf::rocm::gfx942::launch_decode<256,1>(type,device_storage+64,device_decoded+16,count,nullptr)));
    if(schedule==2)check((libgguf::rocm::gfx942::launch_decode<128,2>(type,device_storage+64,device_decoded+16,count,nullptr)));
    if(schedule==3)check((libgguf::rocm::gfx942::launch_decode<256,2>(type,device_storage+64,device_decoded+16,count,nullptr)));
    check(hipDeviceSynchronize());
    check(hipMemcpy(decoded.data(),device_decoded,decoded.size()*sizeof(float),hipMemcpyDeviceToHost));
    for(int i=0;i<16;++i)if(decoded[i]!=-12345 || decoded[count+16+i]!=-12345)throw std::runtime_error("gfx942 decoder pipeline guard failure");
    for(int i=0;i<count;++i)if(!std::isfinite(decoded[16+i]) || std::abs(decoded[16+i]-oracle[i])>1e-5F*std::max(1.0F,std::abs(oracle[i])))throw std::runtime_error("gfx942 decoder pipeline REF mismatch");
  }
  std::vector<float> preserved(count);
  check(hipMemcpy(preserved.data(),device_input,count*sizeof(float),hipMemcpyDeviceToHost));
  if(std::memcmp(preserved.data(),input.data(),count*sizeof(float))) throw std::runtime_error("quantizer modified input storage");
  check(hipMemcpy(preserved.data(),device_importance,count*sizeof(float),hipMemcpyDeviceToHost));
  if(std::memcmp(preserved.data(),importance.data(),count*sizeof(float))) throw std::runtime_error("quantizer modified importance storage");
  check(hipMemcpy(guarded.data()+64,device_storage+64,bytes,hipMemcpyDeviceToHost));
  if(std::memcmp(expected.data(),guarded.data()+64,bytes)) throw std::runtime_error("decoder modified packed input storage");
  check(hipFree(device_input));check(hipFree(device_importance));check(hipFree(device_storage));check(hipFree(device_decoded));
  std::cout << "{\"type\":\"" << libgguf_type_name(type) << "\",\"blocks\":" << blocks
            << ",\"rows\":" << rows << ",\"weighted\":" << (weighted ? "true":"false")
            << ",\"bytes_checked\":" << bytes << ",\"guards_checked\":true,\"status\":\"passed\"}\n";
}

static void benchmark(ggml_type type,int mib) {
  const auto block=libgguf_block_size(type);
  const int count=int((int64_t(mib)*1024*1024/sizeof(float))/block*block);
  if(count<=0) throw std::runtime_error("benchmark allocation must contain at least one block");
  const auto bytes=libgguf_row_size(type,count);
  std::vector<float> input(count),importance(count,1.0F);
  for(int i=0;i<count;++i) input[i]=float(i%127-63)/32;
  float *x=nullptr,*weights=nullptr; void* y=nullptr;
  check(hipMalloc(&x,count*sizeof(float)));check(hipMalloc(&weights,count*sizeof(float)));check(hipMalloc(&y,bytes));
  check(hipMemcpy(x,input.data(),count*sizeof(float),hipMemcpyHostToDevice));
  check(hipMemcpy(weights,importance.data(),count*sizeof(float),hipMemcpyHostToDevice));
  for(int i=0;i<3;++i) quantize_row(x,weights,y,type,count,count,nullptr);
  check(hipDeviceSynchronize());
  hipEvent_t start{},stop{};check(hipEventCreate(&start));check(hipEventCreate(&stop));
  for(int repeat=0;repeat<5;++repeat) {
    check(hipEventRecord(start));
    for(int i=0;i<10;++i) quantize_row(x,weights,y,type,count,count,nullptr);
    check(hipEventRecord(stop));check(hipEventSynchronize(stop));
    float milliseconds=0;check(hipEventElapsedTime(&milliseconds,start,stop));
    std::cout << "{\"type\":\"" << libgguf_type_name(type) << "\",\"input_bytes\":" << count*sizeof(float)
              << ",\"output_bytes\":" << bytes << ",\"importance_bytes\":" << count*sizeof(float)
              << ",\"sample\":" << repeat << ",\"mean_ms\":" << milliseconds/10
              << ",\"working_set\":\"explicit-size\",\"status\":\"measured\"}\n";
  }
  float* decoded=nullptr;check(hipMalloc(&decoded,count*sizeof(float)));
  for(int schedule=0;schedule<4;++schedule) {
    auto enqueue=[&]() {
      if(schedule==0)check((libgguf::rocm::gfx942::launch_decode<128,1,false>(type,y,decoded,count,nullptr)));
      if(schedule==1)check((libgguf::rocm::gfx942::launch_decode<256,1,false>(type,y,decoded,count,nullptr)));
      if(schedule==2)check((libgguf::rocm::gfx942::launch_decode<128,2,false>(type,y,decoded,count,nullptr)));
      if(schedule==3)check((libgguf::rocm::gfx942::launch_decode<256,2,false>(type,y,decoded,count,nullptr)));
    };
    for(int i=0;i<3;++i)enqueue();check(hipDeviceSynchronize());
    for(int repeat=0;repeat<5;++repeat) {
      check(hipEventRecord(start));for(int i=0;i<10;++i)enqueue();
      check(hipEventRecord(stop));check(hipEventSynchronize(stop));
      float milliseconds=0;check(hipEventElapsedTime(&milliseconds,start,stop));
      std::cout<<"{\"type\":\""<<libgguf_type_name(type)<<"\",\"candidate\":\"gfx942.coalesced_decode."<<schedule<<"\",\"sample\":"<<repeat<<",\"mean_ms\":"<<milliseconds/10<<",\"status\":\"measured\"}\n";
    }
  }
  check(hipFree(decoded));
  check(hipEventDestroy(start));check(hipEventDestroy(stop));check(hipFree(x));check(hipFree(weights));check(hipFree(y));
}

int main(int argc,char** argv) {
  try {
    int mib=0;
    std::string format;
    for (int i=1; i<argc; ++i) {
      const std::string option=argv[i];
      if (i+1<argc && option=="--benchmark-mib") mib=std::stoi(argv[++i]);
      else if (i+1<argc && option=="--format") format=argv[++i];
      else throw std::runtime_error("usage: gfx942_quantizer_parity [--format qtype] [--benchmark-mib 256]");
    }
    if(mib<0 || mib>512) throw std::runtime_error("benchmark MiB must be between 1 and 512");
    int ordinal=0;hipDeviceProp_t p{};check(hipGetDevice(&ordinal));check(hipGetDeviceProperties(&p,ordinal));
    if(std::string(p.gcnArchName).substr(0,6)!="gfx942" || p.warpSize!=64) throw std::runtime_error("requires gfx942 wave64");
    const ggml_type types[]={GGML_TYPE_Q1_0,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,
      GGML_TYPE_Q8_0,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,
      GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
      GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS,GGML_TYPE_TQ1_0,
      GGML_TYPE_TQ2_0,GGML_TYPE_MXFP4,GGML_TYPE_NVFP4};
    bool found=false;
    for(auto type:types) {
      if (!format.empty() && format!=libgguf_type_name(type)) continue;
      found=true;
      if(!quantize_type_supported(type) || !dequantize_type_supported(type)) throw std::runtime_error("listed qtype lacks a production GPU implementation");
      for(int blocks:{1,2,3,5}) parity(type,blocks);
      if (type==GGML_TYPE_Q4_0 || type==GGML_TYPE_Q4_1 || type==GGML_TYPE_Q5_0 || type==GGML_TYPE_Q5_1)
        parity(type,129,3,true);
      if(type==GGML_TYPE_Q4_0 || type==GGML_TYPE_Q4_1 || type==GGML_TYPE_Q5_0 ||
         type==GGML_TYPE_Q5_1 || type==GGML_TYPE_Q2_K || type==GGML_TYPE_Q3_K || type==GGML_TYPE_Q4_K ||
         type==GGML_TYPE_Q5_K || type==GGML_TYPE_Q6_K || type==GGML_TYPE_IQ3_S ||
         type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_IQ4_NL || type==GGML_TYPE_IQ4_XS) {
        parity(type,3,4,true);
        parity(type,3,4,false);
      }
    }
    if(!found) throw std::runtime_error("unknown quantization format: "+format);
    if(mib) for(auto type:types)
      if(format.empty() || format==libgguf_type_name(type)) benchmark(type,mib);
    return 0;
  } catch(const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
