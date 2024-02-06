// ======================================================================== //
// Copyright 2020 Ingo Wald                                                 //
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

#include "cuda_helper.h"
#include "deepCompositing.h"
#include "dlfcn.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// std
#include <map>
#include <mutex>
#include <stdexcept>

#undef ANARI_INTERFACE
#define ANARI_INTERFACE extern "C"

#undef ANARI_DEFAULT_VAL
#define ANARI_DEFAULT_VAL(a) /* ignore */

#if 0
#define LOG_API_CALL                                                           \
  {                                                                            \
    fflush(0);                                                                 \
  }
#else
#define LOG_API_CALL
#endif

#include <exception>
#include <limits>
#include <numeric>

using vec2i = anari::math::int2;
using vec3f = float3;

struct PTLib;

struct Frame
{
  Frame(MPI_Comm comm = MPI_COMM_WORLD) : deepComp(1, comm)
  {
    MPI_Comm_rank(comm, &rank);
  }

  void resize(vec2i newSize);
  void composite(PTLib *ptLib, ANARIDevice device, ANARIFrame frame);

  vec2i size = {0, 0};
  std::vector<float> depth;
  std::vector<uint32_t> color;
  uint32_t *d_color_in = 0;
  uint32_t *d_color_out = 0;
  float *d_depth = 0;
  vec2i d_size = {0, 0};

  int rank;
  dc::Compositor deepComp;
};

struct PTLib
{
  std::map<void *, Frame *> frames;
  Frame *findFrame(ANARIFrame frame)
  {
    return findFrame((ANARIObject)frame);
  }
  Frame *findFrame(ANARIObject obj)
  {
    if (frames.find((void *)obj) == frames.end())
      return 0;
    return frames[(void *)obj];
  }

  PTLib()
  {
    const char *ptLibName = "/home/wald/opt/lib/libanari_original.so";
    // const char *ptLibName = "/home/wald/opt/lib/libanari.so";
    lib = dlopen(ptLibName, RTLD_GLOBAL | RTLD_LAZY);
    if (!lib)
      throw std::runtime_error("could not dlopen " + std::string(ptLibName));

    (void *&)loadLibrary = loadSymbol("anariLoadLibrary");
    (void *&)anariNewWorld = loadSymbol("anariNewWorld");
    (void *&)anariCommitParameters = loadSymbol("anariCommitParameters");
    (void *&)anariNewDevice = loadSymbol("anariNewDevice");
    (void *&)anariNewRenderer = loadSymbol("anariNewRenderer");
    (void *&)anariNewFrame = loadSymbol("anariNewFrame");
    (void *&)anariSetParameter = loadSymbol("anariSetParameter");
    (void *&)anariNewCamera = loadSymbol("anariNewCamera");
    (void *&)anariNewMaterial = loadSymbol("anariNewMaterial");
    (void *&)anariNewGeometry = loadSymbol("anariNewGeometry");
    (void *&)anariMapParameterArray1D = loadSymbol("anariMapParameterArray1D");
    (void *&)anariUnmapParameterArray = loadSymbol("anariUnmapParameterArray");
    (void *&)anariNewSurface = loadSymbol("anariNewSurface");
    (void *&)anariRelease = loadSymbol("anariRelease");
    (void *&)anariNewGroup = loadSymbol("anariNewGroup");
    (void *&)anariNewInstance = loadSymbol("anariNewInstance");
    (void *&)anariNewArray1D = loadSymbol("anariNewArray1D");
    (void *&)anariRenderFrame = loadSymbol("anariRenderFrame");
    (void *&)anariMapFrame = loadSymbol("anariMapFrame");
    // xx
  }

  const void *(*anariMapFrame)(ANARIDevice device,
      ANARIFrame frame,
      const char *channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType);
  void (*anariRenderFrame)(ANARIDevice dev, ANARIFrame frame);
  ANARIArray1D (*anariNewArray1D)(ANARIDevice device,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userData,
      ANARIDataType dataType,
      uint64_t numElements);
  ANARIInstance (*anariNewInstance)(ANARIDevice device, const char *type);
  ANARIGroup (*anariNewGroup)(ANARIDevice dev);
  void (*anariRelease)(ANARIDevice dev, ANARIObject obj);
  ANARISurface (*anariNewSurface)(ANARIDevice dev);
  void (*anariUnmapParameterArray)(
      ANARIDevice device, ANARIObject object, const char *name);
  void *(*anariMapParameterArray1D)(ANARIDevice device,
      ANARIObject object,
      const char *name,
      ANARIDataType dataType,
      uint64_t numElements,
      uint64_t *elementStride);
  ANARIGeometry (*anariNewGeometry)(ANARIDevice dev, const char *type);
  ANARIMaterial (*anariNewMaterial)(ANARIDevice dev, const char *materialType);
  ANARIRenderer (*anariNewRenderer)(ANARIDevice dev, const char *type);
  ANARILibrary (*loadLibrary)(const char *name,
      ANARIStatusCallback statusCallback,
      const void *statusCallbackUserData);
  ANARIWorld (*anariNewWorld)(ANARIDevice device);
  void (*anariCommitParameters)(ANARIDevice device, ANARIObject object);
  ANARIDevice (*anariNewDevice)(ANARILibrary lib, const char *deviceType);
  ANARIFrame (*anariNewFrame)(ANARIDevice dev);
  void (*anariSetParameter)(ANARIDevice _device,
      ANARIObject _object,
      const char *paramName,
      ANARIDataType type,
      const void *mem);
  ANARICamera (*anariNewCamera)(ANARIDevice dev, const char *type);

  void *loadSymbol(const char *name)
  {
    void *sym = dlsym(lib, name);
    if (!sym)
      throw std::runtime_error("could not dlsym " + std::string(name));
    return sym;
  }

  // std::string libName;
  void *lib;
};
PTLib *ptLib = 0;

__global__ void writeFrags(
    dc::DeviceInterface di, vec2i size, float *d_depth, uint32_t *d_color)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix >= size.x)
    return;
  if (iy >= size.y)
    return;

  float z = d_depth[ix + iy * size.x];
  uint32_t rgba = d_color[ix + iy * size.x];

  if (ix == 512 && iy == 400)
    printf("z is %f\n", z);

  float a = ((rgba >> 24) & 0xff) / 255.f;
  float b = ((rgba >> 16) & 0xff) / 255.f;
  float g = ((rgba >> 8) & 0xff) / 255.f;
  float r = ((rgba >> 0) & 0xff) / 255.f;
  dc::Fragment frag(z, make_float3(r, g, b), a);
  di.write(make_int2(ix, iy), frag);
}

void Frame::composite(PTLib *ptLib, ANARIDevice device, ANARIFrame frame)
{
  dc::DeviceInterface dcDev = deepComp.prepare();

  uint32_t ptSize[2];
  ANARIDataType ptType;
  const float *depth = (const float *)ptLib->anariMapFrame(
      device, frame, "channel.depth", &ptSize[0], &ptSize[1], &ptType);
  const uint32_t *color = (const uint32_t *)ptLib->anariMapFrame(
      device, frame, "channel.color", &ptSize[0], &ptSize[1], &ptType);
  CUDA_SYNC_CHECK();
  cudaMemcpy(
      d_depth, depth, size.x * size.y * sizeof(float), cudaMemcpyDefault);
  CUDA_SYNC_CHECK();
  cudaMemcpy(
      d_color_in, color, size.x * size.y * sizeof(uint32_t), cudaMemcpyDefault);
  CUDA_SYNC_CHECK();

  writeFrags<<<dc::divRoundUp(vec2i(size), vec2i(16)), vec2i(16)>>>(
      deepComp.prepare(), size, d_depth, d_color_in);
  deepComp.finish(rank == 0 ? d_color_out : nullptr);
  cudaMemcpy(this->color.data(),
      d_color_out,
      size.x * size.y * sizeof(uint32_t),
      cudaMemcpyDefault);
  CUDA_SYNC_CHECK();
}

void Frame::resize(dc::vec2i newSize)
{
  if (newSize.x * newSize.y == 0)
    return;
  if (newSize == size)
    return;

  size = d_size = newSize;
  deepComp.resize((const int2 &)size);
  color.resize(size.x * size.y);
  depth.resize(size.x * size.y);

  if (d_depth)
    cudaFree(d_depth);
  if (d_color_in)
    cudaFree(d_color_in);
  if (d_color_out)
    cudaFree(d_color_out);
  CUDA_SYNC_CHECK();

  cudaMalloc((void **)&d_color_in, size.x * size.y * sizeof(*d_color_in));
  CUDA_SYNC_CHECK();
  cudaMalloc((void **)&d_color_out, size.x * size.y * sizeof(*d_color_out));
  CUDA_SYNC_CHECK();
  cudaMalloc((void **)&d_depth, size.x * size.y * sizeof(*d_depth));
  CUDA_SYNC_CHECK();
}

#define NOT_IMPLEMENTED                                                        \
  {                                                                            \
    std::cerr << " NOT IMPLEMENTED : ";                                        \
    exit(1);                                                                   \
  }

#define THROW_IF_NULL(obj, name)                                               \
  if (obj == nullptr)                                                          \
  throw std::runtime_error(std::string("null ") + name                         \
      + std::string(" provided to ") + __FUNCTION__)

// convenience macros for repeated use of the above
#define THROW_IF_NULL_OBJECT(obj) THROW_IF_NULL(obj, "handle")
#define THROW_IF_NULL_STRING(str) THROW_IF_NULL(str, "string")

#define ASSERT_DEVICE()                                                        \
  if (!deviceIsSet())                                                          \
  throw std::runtime_error(                                                    \
      "ANARI not yet initialized "                                             \
      "(most likely this means you tried to "                                  \
      "call an anari API function before "                                     \
      "first calling anariInit())")

#define ANARI_CATCH_BEGIN try {
#define ANARI_CATCH_END(a)                                                     \
  }                                                                            \
  catch (const std::bad_alloc &)                                               \
  {                                                                            \
    /* TODO: handle error */                                                   \
    return a;                                                                  \
  }                                                                            \
  catch (const std::exception &e)                                              \
  {                                                                            \
    /* TODO: handle error */                                                   \
    std::cerr << "FATAL ERROR " << e.what() << std::endl;                      \
    exit(1);                                                                   \
    return a;                                                                  \
  }                                                                            \
  catch (...)                                                                  \
  {                                                                            \
    /* TODO: handle error */                                                   \
    return a;                                                                  \
  }

// using anari::Device;

///////////////////////////////////////////////////////////////////////////////
// Initialization /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::mutex anari_g_mutex;
#define LOCK_GUARD() std::lock_guard<std::mutex> lock(anari_g_mutex);

// ==================================================================
ANARI_INTERFACE ANARIFrame anariNewFrame(ANARIDevice dev) ANARI_CATCH_BEGIN
{
  LOG_API_CALL;
  LOCK_GUARD();
  THROW_IF_NULL_OBJECT(dev);
#if 1
  ANARIFrame frame = ptLib->anariNewFrame(dev);
  ptLib->frames[frame] = new Frame;
  return frame;
#else
  return ptLib->anariNewFrame(dev);
#endif
}
ANARI_CATCH_END(nullptr)
// ==================================================================

// Pointer access (read-only) to the memory of the given frame buffer channel
ANARI_INTERFACE
const void *anariMapFrame(ANARIDevice device,
    ANARIFrame _frame,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType) ANARI_CATCH_BEGIN
{
#if 1
  Frame *frame = ptLib->findFrame(_frame);
  *width = frame->size.x;
  *height = frame->size.y;
  if (0 == strcmp(channel, "channel.color")) {
    *pixelType = ANARI_UFIXED8_VEC4;
    return frame->color.data();
  }
  if (channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return frame->depth.data();
  }

  NOT_IMPLEMENTED;
#else
  return ptLib->anariMapFrame(
      device, _frame, channel, width, height, pixelType);
#endif
}
ANARI_CATCH_END(0)
// ==================================================================

// Unmap a previously mapped frame buffer pointer
ANARI_INTERFACE
void anariUnmapFrame(
    ANARIDevice device, ANARIFrame frame, const char *channel) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END()

// ==================================================================
ANARI_INTERFACE void anariCommitParameters(
    ANARIDevice device, ANARIObject object)
    // ANARI_INTERFACE void anariCommit(ANARIDevice _device, ANARIObject
    // _object)
    ANARI_CATCH_BEGIN
{
  LOG_API_CALL;
  LOCK_GUARD();
  THROW_IF_NULL_OBJECT(object);
  THROW_IF_NULL_OBJECT(object);

  ptLib->anariCommitParameters(device, object);

  // if (object == (ANARIObject)self) {
  //   self->commit();
  //   return;
  // }
  // // auto *device = (ao::Device *)_device;
  // // device->commit(_object);
  // NOT_IMPLEMENTED;
}
ANARI_CATCH_END()

// ==================================================================
// Set a parameter, where 'mem' points the address of the type specified
ANARI_INTERFACE void anariSetParameter(ANARIDevice _device,
    ANARIObject _object,
    const char *paramName,
    ANARIDataType type,
    const void *mem) ANARI_CATCH_BEGIN
{
  // anari::setParameter(device, frame, "size", (const
  // anari::math::uint2&)fbSize); anari::setParameter(device, frame,
  // "channel.color", ANARI_UFIXED8_VEC4);
  Frame *frame = ptLib->findFrame(_object);
  if (frame) {
    if (0 == strcmp(paramName, "size"))
      frame->resize(*(const vec2i *)mem);
  }
  ptLib->anariSetParameter(_device, _object, paramName, type, mem);
}
ANARI_CATCH_END()
// ==================================================================

// Reduce the application-side object ref count by 1
ANARI_INTERFACE
void anariRelease(ANARIDevice dev, ANARIObject obj) ANARI_CATCH_BEGIN
{
  ptLib->anariRelease(dev, obj);
}
ANARI_CATCH_END()
// ==================================================================

// Increace the application-side object ref count by 1
ANARI_INTERFACE void anariRetain(ANARIDevice, ANARIObject) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END()
// ==================================================================

ANARI_INTERFACE
void *anariMapParameterArray1D(ANARIDevice device,
    ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride) ANARI_CATCH_BEGIN
{
  return ptLib->anariMapParameterArray1D(
      device, object, name, dataType, numElements1, elementStride);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE void *anariMapParameterArray2D(ANARIDevice device,
    ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE void *anariMapParameterArray3D(ANARIDevice device,
    ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE
void anariUnmapParameterArray(
    ANARIDevice device, ANARIObject object, const char *name) ANARI_CATCH_BEGIN
{
  ptLib->anariUnmapParameterArray(device, object, name);
}
ANARI_CATCH_END()
// ==================================================================

ANARI_INTERFACE
ANARIArray1D anariNewArray1D(ANARIDevice device,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType dataType,
    uint64_t numElements1) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewArray1D(
      device, appMemory, deleter, userData, dataType, numElements1);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARIArray2D anariNewArray2D(ANARIDevice device,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2)
    // ANARI_INTERFACE ANARIArray2D anariNewArray2D(ANARIDevice,
    //     void *appMemory,
    //     ANARIMemoryDeleter,
    //     void *userPtr,
    //     ANARIDataType,
    //     uint64_t numItems1,
    //     uint64_t numItems2,
    //     uint64_t byteStride1 ANARI_DEFAULT_VAL(0),
    //     uint64_t byteStride2 ANARI_DEFAULT_VAL(0))
    ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARIArray3D anariNewArray3D(ANARIDevice device,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3)
    // ANARI_INTERFACE ANARIArray3D anariNewArray3D(ANARIDevice,
    //     void *appMemory,
    //     ANARIMemoryDeleter,
    //     void *userPtr,
    //     ANARIDataType,
    //     uint64_t numItems1,
    //     uint64_t numItems2,
    //     uint64_t numItems3,
    //     uint64_t byteStride1 ANARI_DEFAULT_VAL(0),
    //     uint64_t byteStride2 ANARI_DEFAULT_VAL(0),
    //     uint64_t byteStride3 ANARI_DEFAULT_VAL(0))
    ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARIMaterial anariNewMaterial(
    ANARIDevice dev, const char *materialType) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewMaterial(dev, materialType);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE int anariGetProperty(ANARIDevice,
    ANARIObject,
    const char *propertyName,
    ANARIDataType propertyType,
    void *memory,
    uint64_t size,
    ANARIWaitMask) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE
ANARISurface anariNewSurface(ANARIDevice dev) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewSurface(dev);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARIRenderer anariNewRenderer(
    ANARIDevice dev, const char *type) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewRenderer(dev, type);
}
ANARI_CATCH_END(0)

// ==================================================================
ANARI_INTERFACE ANARIGeometry anariNewGeometry(
    ANARIDevice dev, const char *type) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewGeometry(dev, type);
}
ANARI_CATCH_END(0)
// ==================================================================

// Render a frame (non-blocking)
ANARI_INTERFACE
void anariRenderFrame(ANARIDevice dev, ANARIFrame frame) ANARI_CATCH_BEGIN
{
  ptLib->anariRenderFrame(dev, frame);
  ptLib->findFrame(frame)->composite(ptLib, dev, frame);
}
ANARI_CATCH_END()
// ==================================================================

// Ask if the frame is ready (ANARI_NO_WAIT) or wait until ready (ANARI_WAIT)
ANARI_INTERFACE int anariFrameReady(
    ANARIDevice, ANARIFrame, ANARIWaitMask) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

// Signal the running frame should be canceled if possible
ANARI_INTERFACE void anariDiscardFrame(
    ANARIDevice, ANARIFrame) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END()
// ==================================================================

// Query whether a device supports an extension
ANARI_INTERFACE int anariDeviceImplements(
    ANARIDevice, const char *extension) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

// Create an ANARI engine backend using explicit device string.
ANARI_INTERFACE
ANARIDevice anariNewDevice(ANARILibrary lib,
    const char *deviceType ANARI_DEFAULT_VAL("default")) ANARI_CATCH_BEGIN
{
  if (!ptLib)
    throw std::runtime_error("library not yet loaded...");
  return ptLib->anariNewDevice(lib, deviceType);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE
ANARIGroup anariNewGroup(ANARIDevice dev) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewGroup(dev);
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE
ANARIInstance anariNewInstance(ANARIDevice device, const char *type)
    // ANARI_INTERFACE ANARIInstance anariNewInstance(ANARIDevice)
    ANARI_CATCH_BEGIN
{
  return ptLib->anariNewInstance(device, type);
}
ANARI_CATCH_END(0)
// ==================================================================

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARI_INTERFACE
ANARIWorld anariNewWorld(ANARIDevice device) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewWorld(device);
  // NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

// Extension Objects //////////////////////////////////////////////////////////

ANARI_INTERFACE ANARIObject anariNewObject(
    ANARIDevice, const char *objectType, const char *type) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARISpatialField anariNewSpatialField(
    ANARIDevice, const char *type) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARILight anariNewLight(
    ANARIDevice, const char *type) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARICamera anariNewCamera(
    ANARIDevice dev, const char *type) ANARI_CATCH_BEGIN
{
  return ptLib->anariNewCamera(dev, type);
}
ANARI_CATCH_END(0)
// ==================================================================

/* Load library 'name' from shared lib libanari_library_<name>.so
   errors are reported by invoking the passed ANARIStatusCallback if it is
   not NULL */
ANARI_INTERFACE
ANARILibrary anariLoadLibrary(const char *name,
    ANARIStatusCallback statusCallback ANARI_DEFAULT_VAL(0),
    const void *statusCallbackUserData ANARI_DEFAULT_VAL(0))
    // ANARI_INTERFACE ANARILibrary anariLoadLibrary(const char *name,
    //     ANARIStatusCallback defaultStatusCB ANARI_DEFAULT_VAL(NULL),
    //     void *defaultStatusCBUserPtr ANARI_DEFAULT_VAL(NULL))
    ANARI_CATCH_BEGIN
{
  // std::cout << "#ao: supposed to load a library '" << name << "'" <<
  // std::endl; std::string libName =
  // "libanari_library_"+std::string(name)+".so"; std::string libName =
  // "/home/wald/opt/lib/libanari_library_"+std::string(name)+".so";
  if (ptLib)
    throw std::runtime_error("library already loaded...");
  ptLib = new PTLib;
  return ptLib->loadLibrary(name, statusCallback, statusCallbackUserData);

  // return (ANARILibrary)ptLib;

  // NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE void anariUnloadLibrary(ANARILibrary) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END()
// ==================================================================

ANARI_INTERFACE ANARISampler anariNewSampler(
    ANARIDevice, const char *type) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================

ANARI_INTERFACE ANARIVolume anariNewVolume(
    ANARIDevice, const char *type) ANARI_CATCH_BEGIN
{
  NOT_IMPLEMENTED;
}
ANARI_CATCH_END(0)
// ==================================================================
