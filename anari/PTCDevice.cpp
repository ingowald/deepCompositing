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

#include "PTCDevice.h"
// helium
#include <helium/BaseGlobalDeviceState.h>
#include <helium/BaseObject.h>
// std
#include <memory>

namespace ptc {

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D PTCDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();
  return m_ptd->newArray1D(appMemory, deleter, userData, type, numItems);
}

ANARIArray2D PTCDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();
  return m_ptd->newArray2D(
      appMemory, deleter, userData, type, numItems1, numItems2);
}

ANARIArray3D PTCDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();
  return m_ptd->newArray3D(
      appMemory, deleter, userData, type, numItems1, numItems2, numItems3);
}

ANARICamera PTCDevice::newCamera(const char *subtype)
{
  initDevice();
  return m_ptd->newCamera(subtype);
}

ANARIFrame PTCDevice::newFrame()
{
  initDevice();
  return m_ptd->newFrame();
}

ANARIGeometry PTCDevice::newGeometry(const char *subtype)
{
  initDevice();
  return m_ptd->newGeometry(subtype);
}

ANARIGroup PTCDevice::newGroup()
{
  initDevice();
  return m_ptd->newGroup();
}

ANARIInstance PTCDevice::newInstance(const char *subtype)
{
  initDevice();
  return m_ptd->newInstance(subtype);
}

ANARILight PTCDevice::newLight(const char *subtype)
{
  initDevice();
  return m_ptd->newLight(subtype);
}

ANARIMaterial PTCDevice::newMaterial(const char *subtype)
{
  initDevice();
  return m_ptd->newMaterial(subtype);
}

ANARIRenderer PTCDevice::newRenderer(const char *subtype)
{
  initDevice();
  return m_ptd->newRenderer(subtype);
}

ANARISampler PTCDevice::newSampler(const char *subtype)
{
  initDevice();
  return m_ptd->newSampler(subtype);
}

ANARISpatialField PTCDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return m_ptd->newSpatialField(subtype);
}

ANARISurface PTCDevice::newSurface()
{
  initDevice();
  return m_ptd->newSurface();
}

ANARIVolume PTCDevice::newVolume(const char *subtype)
{
  initDevice();
  return m_ptd->newVolume(subtype);
}

ANARIWorld PTCDevice::newWorld()
{
  initDevice();
  return m_ptd->newWorld();
}

// Object + Parameter Lifetime Management /////////////////////////////////////

void PTCDevice::setParameter(
    ANARIObject object, const char *name, ANARIDataType type, const void *mem)
{
  if (handleIsDevice(object))
    this->setParam(name, type, mem);
  else
    m_ptd->setParameter(object, name, type, mem);
}

void PTCDevice::unsetParameter(ANARIObject object, const char *name)
{
  if (handleIsDevice(object))
    this->removeParam(name);
  else
    m_ptd->unsetParameter(object, name);
}

void PTCDevice::unsetAllParameters(ANARIObject object)
{
  if (handleIsDevice(object))
    this->removeAllParams();
  else
    m_ptd->unsetAllParameters(object);
}

void *PTCDevice::mapParameterArray1D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  return m_ptd->mapParameterArray1D(
      object, name, dataType, numElements1, elementStride);
}

void *PTCDevice::mapParameterArray2D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  return m_ptd->mapParameterArray2D(
      object, name, dataType, numElements1, numElements2, elementStride);
}

void *PTCDevice::mapParameterArray3D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  return m_ptd->mapParameterArray3D(object,
      name,
      dataType,
      numElements1,
      numElements2,
      numElements3,
      elementStride);
}

void PTCDevice::unmapParameterArray(ANARIObject object, const char *name)
{
  m_ptd->unmapParameterArray(object, name);
}

void PTCDevice::commitParameters(ANARIObject object)
{
  if (handleIsDevice(object))
    deviceCommitParameters();
  else
    m_ptd->commitParameters(object);
}

void PTCDevice::release(ANARIObject o)
{
  if (o == nullptr)
    return;

  if (handleIsDevice(o)) {
    if (--m_refCount == 0)
      delete this;
    return;
  }

  m_ptd->release(o);
}

void PTCDevice::retain(ANARIObject o)
{
  if (handleIsDevice(o))
    m_refCount++;
  else
    m_ptd->retain(o);
}

void *PTCDevice::mapArray(ANARIArray a)
{
  return m_ptd->mapArray(a);
}

void PTCDevice::unmapArray(ANARIArray a)
{
  m_ptd->unmapArray(a);
}

int PTCDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  return m_ptd->getProperty(object, name, type, mem, size, mask);
}

// Query functions ////////////////////////////////////////////////////////////

const char **PTCDevice::getObjectSubtypes(ANARIDataType objectType)
{
  initDevice();
  return m_ptd->getObjectSubtypes(objectType);
}

const void *PTCDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  initDevice();
  return m_ptd->getObjectInfo(objectType, objectSubtype, infoName, infoType);
}

const void *PTCDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  initDevice();
  return m_ptd->getParameterInfo(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// FrameBuffer Manipulation ///////////////////////////////////////////////////

const void *PTCDevice::frameBufferMap(ANARIFrame f,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  return m_ptd->frameBufferMap(f, channel, width, height, pixelType);
}

void PTCDevice::frameBufferUnmap(ANARIFrame f, const char *channel)
{
  m_ptd->frameBufferUnmap(f, channel);
}

// Frame Rendering ////////////////////////////////////////////////////////////

void PTCDevice::renderFrame(ANARIFrame f)
{
  m_ptd->renderFrame(f);
}

int PTCDevice::frameReady(ANARIFrame f, ANARIWaitMask m)
{
  return m_ptd->frameReady(f, m);
}

void PTCDevice::discardFrame(ANARIFrame f)
{
  m_ptd->discardFrame(f);
}

// Other PTCDevice definitions ////////////////////////////////////////////////

PTCDevice::PTCDevice(ANARIStatusCallback defaultCallback, const void *userPtr)
{
  anari::DeviceImpl::m_defaultStatusCB = defaultCallback;
  anari::DeviceImpl::m_defaultStatusCBUserPtr = userPtr;
}

PTCDevice::PTCDevice(ANARILibrary l) : DeviceImpl(l) {}

PTCDevice::~PTCDevice()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying PTC device (%p)", this);
  if (m_ptd)
    m_ptd->release(m_ptd->this_device());
}

void PTCDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing PTC device (%p)", this);
  ANARILibrary lib = anariLoadLibrary(
      "environment", defaultStatusCallback(), defaultStatusCallbackUserPtr());
  if (!lib) {
    reportMessage(ANARI_SEVERITY_WARNING, "falling back to 'helide'");
    lib = anariLoadLibrary(
        "helide", defaultStatusCallback(), defaultStatusCallbackUserPtr());
  }

  if (!lib) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to load passthrough lib");
    return;
  }

  m_ptd = (anari::DeviceImpl *)anariNewDevice(lib, "default");

  if (!m_ptd)
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to create device");

  m_initialized = true;
}

void PTCDevice::deviceCommitParameters()
{
  // TODO
}

int PTCDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  return 0;
}

} // namespace ptc
