// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>
#include "core/framework/allocator.h"
#include <mutex>

namespace onnxruntime {

class MIGraphXAllocator : public IAllocator {
 public:
  MIGraphXAllocator(int device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD,
                                    static_cast<OrtDevice::DeviceId>(device_id)),
                          OrtMemTypeDefault)) {}

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;

 private:
  void CheckDevice() const;
};

class MIGraphXExternalAllocator : public MIGraphXAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  MIGraphXExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : MIGraphXAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
    empty_cache_ = reinterpret_cast<ExternalEmptyCache>(empty_cache);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

 private:
  mutable std::mutex lock_;
  ExternalAlloc alloc_;
  ExternalFree free_;
  ExternalEmptyCache empty_cache_;
  std::unordered_set<void*> reserved_;
};

class MIGraphXPinnedAllocator final : public IAllocator {
 public:
  MIGraphXPinnedAllocator(const int device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::AMD,
                                    static_cast<OrtDevice::DeviceId>(device_id)),
                          OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

}  // namespace onnxruntime
