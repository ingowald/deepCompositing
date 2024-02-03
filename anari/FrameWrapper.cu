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
}

FrameWrapper::~FrameWrapper()
{
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
    m_size = bit_cast<anari::math::uint2>(type, mem);

  anariSetParameter(m_device, m_frame, _name, type, mem);
}

void FrameWrapper::unsetParameter(const char *name)
{
  anariUnsetParameter(m_device, m_frame, name);
}

void FrameWrapper::unsetAllParameters()
{
  anariUnsetAllParameters(m_device, m_frame);
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

const void *FrameWrapper::frameBufferMap(const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  return anariMapFrame(m_device, m_frame, channel, width, height, pixelType);
}

void FrameWrapper::frameBufferUnmap(const char *channel)
{
  anariUnmapFrame(m_device, m_frame, channel);
}

void FrameWrapper::renderFrame()
{
  anariRenderFrame(m_device, m_frame);
}

int FrameWrapper::frameReady(ANARIWaitMask m)
{
  return anariFrameReady(m_device, m_frame, m);
}

void FrameWrapper::discardFrame()
{
  anariDiscardFrame(m_device, m_frame);
}

} // namespace ptc