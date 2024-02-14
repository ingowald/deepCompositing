// ======================================================================== //
// Copyright 2024 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "FrameWrapper.h"
// std
#include <cstring>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

#include "../cuda_helper.h"

namespace ptc {

// Helper functions ///////////////////////////////////////////////////////////

template <typename T>
T bit_cast(ANARIDataType type, const void *mem)
{
  if (sizeof(T) != anari::sizeOf(type))
    throw std::runtime_error("T size mismatch");
  T retval;
  std::memcpy(&retval, mem, sizeof(T));
  return retval;
}

// GPU kernels ////////////////////////////////////////////////////////////////

__global__ void writeFrags(dc::DeviceInterface di,
    uint32_t size_x,
    uint32_t size_y,
    const float *d_depth,
    const void *d_color,
    ANARIDataType colorType)
{
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix >= size_x)
    return;
  if (iy >= size_y)
    return;

  const auto offset = ix + iy * size_x;
  const float z = d_depth[offset];

#if 0
  if (ix == 512 && iy == 400)
    printf("z is %f\n", z);
#endif

  if (colorType == ANARI_FLOAT32_VEC4) {
    const float *rgba = (const float *)d_color + (offset * 4);
    const float a = rgba[0];
    const float b = rgba[1];
    const float g = rgba[2];
    const float r = rgba[3];
    dc::Fragment frag(z, make_float3(r, g, b), a);
    di.write(ix, iy, frag);
  } else {
    const uint32_t rgba = *((const uint32_t *)d_color + offset);
    const float a = ((rgba >> 24) & 0xff) / 255.f;
    const float b = ((rgba >> 16) & 0xff) / 255.f;
    const float g = ((rgba >> 8) & 0xff) / 255.f;
    const float r = ((rgba >> 0) & 0xff) / 255.f;
    dc::Fragment frag(z, make_float3(r, g, b), a);
    di.write(ix, iy, frag);
  }
}

// FrameWrapper definitions ///////////////////////////////////////////////////

FrameWrapper::FrameWrapper(ANARIDevice d,
    ANARIFrame f,
    FrameWrapperNotificationHandler onObjectDestroy,
    MPI_Comm comm)
    : m_onObjectDestroy(onObjectDestroy),
      m_deepComp(1, comm),
      m_device(d),
      m_frame(f)
{
  MPI_Comm_rank(comm, &m_rank);
  anariRetain(m_device, m_device);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
  m_deepComp.affinitizeGPU();
}

FrameWrapper::~FrameWrapper()
{
  cleanup();
  anariRelease(m_device, m_frame);
  anariRelease(m_device, m_device);
  m_onObjectDestroy(this);
}

ANARIFrame FrameWrapper::handle() const
{
  return m_frame;
}

void FrameWrapper::setParameter(
    const char *_name, ANARIDataType type, const void *mem)
{
  std::string_view name = _name;

  if (type == ANARI_UINT32_VEC2 && name == "size")
    m_newSize = bit_cast<anari::math::uint2>(type, mem);
  else if (type == ANARI_DATA_TYPE && name == "channel.color")
    m_newColorType = bit_cast<ANARIDataType>(type, mem);
  else if (type == ANARI_DATA_TYPE && name == "channel.depth")
    return; // we don't want the app to turn off the depth channel

  anariSetParameter(m_device, m_frame, _name, type, mem);
}

void FrameWrapper::unsetParameter(const char *name)
{
  anariUnsetParameter(m_device, m_frame, name);
}

void FrameWrapper::unsetAllParameters()
{
  anariUnsetAllParameters(m_device, m_frame);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

void *FrameWrapper::mapParameterArray1D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  return anariMapParameterArray1D(
      m_device, m_frame, name, dataType, numElements1, elementStride);
}

void *FrameWrapper::mapParameterArray2D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  return anariMapParameterArray2D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      elementStride);
}

void *FrameWrapper::mapParameterArray3D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  return anariMapParameterArray3D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      numElements3,
      elementStride);
}

void FrameWrapper::unmapParameterArray(const char *name)
{
  anariUnmapParameterArray(m_device, m_frame, name);
}

void FrameWrapper::commitParameters()
{
  updateSize();
  anariCommitParameters(m_device, m_frame);
}

void FrameWrapper::release()
{
  refDec();
}

void FrameWrapper::retain()
{
  refInc();
}

int FrameWrapper::getProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  return anariGetProperty(m_device, m_frame, name, type, mem, size, mask);
}

const void *FrameWrapper::frameBufferMap(const char *_channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  *width = m_currentSize.x;
  *height = m_currentSize.y;

  std::string_view channel = _channel;

  if (channel == "channel.color") {
    *pixelType = m_currentColorType;
    return m_color.data();
  } else if (channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return m_depth.data();
  }

  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void FrameWrapper::frameBufferUnmap(const char *channel)
{
  // no-op
}

void FrameWrapper::renderFrame()
{
  anariRenderFrame(m_device, m_frame);
  composite();
}

int FrameWrapper::frameReady(ANARIWaitMask m)
{
  return anariFrameReady(m_device, m_frame, m);
}

void FrameWrapper::discardFrame()
{
  anariDiscardFrame(m_device, m_frame);
}

void FrameWrapper::updateSize()
{
  if (m_newSize == m_currentSize)
    return;

  cleanup();

  m_currentSize = m_newSize;
  m_currentColorType = m_newColorType;

  if (m_currentSize.x * m_currentSize.y == 0)
    return;

  if (m_currentColorType == ANARI_UNKNOWN)
    return;

  if (m_currentColorType == ANARI_FLOAT32_VEC4) {
    throw std::runtime_error(
        "support for FLOAT32_VEC4 color channel not implemented");
  }

  const auto &size = m_currentSize;

  m_deepComp.resize(size.x, size.y);
  m_color.resize(size.x * size.y * sizeof(anari::math::float4));
  m_depth.resize(size.x * size.y);

  cudaMalloc((void **)&d_color_in, size.x * size.y * sizeof(*d_color_in));
  cudaMalloc((void **)&d_color_out, size.x * size.y * sizeof(*d_color_out));
  cudaMalloc((void **)&d_depth, size.x * size.y * sizeof(*d_depth));
  CUDA_SYNC_CHECK();
}

void FrameWrapper::composite()
{
  dc::DeviceInterface dcDev = m_deepComp.prepare();

  anari::math::uint2 size;
  ANARIDataType ptType = ANARI_UNKNOWN;
  const float *depth = (const float *)anariMapFrame(
      m_device, m_frame, "channel.depth", &size.x, &size.y, &ptType);
  const uint32_t *color = (const uint32_t *)anariMapFrame(
      m_device, m_frame, "channel.color", &size.x, &size.y, &ptType);
  cudaMemcpy(
      d_depth, depth, size.x * size.y * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(
      d_color_in, color, size.x * size.y * sizeof(uint32_t), cudaMemcpyDefault);

  auto ngx = dc::divRoundUp(size.x, 16);
  auto ngy = dc::divRoundUp(size.y, 16);

  writeFrags<<<dim3(ngx, ngy), dim3(16, 16)>>>(m_deepComp.prepare(),
      size.x,
      size.y,
      d_depth,
      d_color_in,
      m_currentColorType);
  m_deepComp.finish(m_rank == 0 ? d_color_out : nullptr);
  cudaMemcpy(m_color.data(),
      d_color_out,
      size.x * size.y * sizeof(uint32_t),
      cudaMemcpyDefault);

  CUDA_SYNC_CHECK();

  anariUnmapFrame(m_device, m_frame, "channel.depth");
  anariUnmapFrame(m_device, m_frame, "channel.color");
}

void FrameWrapper::cleanup()
{
  if (d_depth)
    cudaFree(d_depth);
  if (d_color_in)
    cudaFree(d_color_in);
  if (d_color_out)
    cudaFree(d_color_out);

  CUDA_SYNC_CHECK();

  d_depth = nullptr;
  d_color_in = nullptr;
  d_color_out = nullptr;
}

} // namespace ptc