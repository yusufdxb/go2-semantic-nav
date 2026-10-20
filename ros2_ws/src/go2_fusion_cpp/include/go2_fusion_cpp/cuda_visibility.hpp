#pragma once
#include "go2_fusion_cpp/core.hpp"
#include <memory>

namespace go2_fusion {
// Optional accelerator. Construction can throw if CUDA is unavailable; callers
// must retain the CPU backend. Instances have one owner and are not thread safe.
class CudaVisibility {
 public:
  enum class TransferMode { ExplicitCopy, MappedPinned };
  struct Stats { double total_ms=0,upload_ms=0,kernels_ms=0,download_ms=0; };
  explicit CudaVisibility(Config config={},TransferMode mode=TransferMode::ExplicitCopy);
  ~CudaVisibility();
  CudaVisibility(const CudaVisibility&)=delete;
  CudaVisibility& operator=(const CudaVisibility&)=delete;
  bool run(const std::vector<Projected>& points,uint32_t width,uint32_t height,
           bool inject_failure=false);
  const uint8_t* flags() const;
  size_t size() const;
  size_t allocated_bytes() const;
  const std::string& error() const;
  const Stats& stats() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace go2_fusion
