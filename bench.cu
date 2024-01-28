// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
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

#include <cuda_runtime.h>
#include <mpi.h>
#include "testRenderer.h"
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

namespace dc {

  __global__ void replayKernel(dc::DeviceInterface dci,
                               const dc::SavedFrags replay,
                               int numPixels)
  {
    int pixelID = threadIdx.x+blockIdx.x*blockDim.x;
    if (pixelID >= numPixels) return;
    int end = replay.counters[pixelID];
    int begin
      = (pixelID == 0)
      ? 0
      : replay.counters[pixelID-1];
    int pixelX = pixelID % replay.fbSize.x;
    int pixelY = pixelID / replay.fbSize.x;
    for (int i=begin;i<end;i++)
      dci.write({pixelX,pixelY},replay.fragments[i]);
  }
                   
                               
                               
  void renderFrame(dc::Compositor &compositor,
                   const dc::SavedFrags &replay,
                   uint32_t *finalPixels)
  {
    dc::DeviceInterface interface = compositor.prepare();

    int numPixels = replay.fbSize.x * replay.fbSize.y;
    int blockSize = 128;
    int numBlocks = divRoundUp(numPixels,blockSize);
    replayKernel<<<numBlocks,blockSize>>>(interface,replay,numPixels);
    
    compositor.finish(finalPixels);
  }

  void runBench(int ac, char **av)
  {
    /* initialize MPI */
    MPI_Init(&ac,&av);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (ac != 2)
      throw std::runtime_error("dcBench <pathToDumpedFiles>");
    std::string fileBase = av[1];
    std::string myFileName = fileBase+"_"+std::to_string(rank)+".dcfb";
    dc::SavedFrags replay = dc::SavedFrags::load(myFileName);
    
    /* create our deep compositor */
    dc::Compositor compositor;
    compositor.affinitizeGPU();
    cudaSetDevice(compositor.affinitizedGPU);

    
    compositor.resize(replay.fbSize.x,replay.fbSize.y);
    const vec2i &fbSize = replay.fbSize;
    
    uint32_t *finalPixels = 0;
    if (rank == 0)
      cudaMalloc(&finalPixels,fbSize.x*fbSize.y*sizeof(*finalPixels));
    
    
    // ------------------------------------------------------------------
    // render one frame
    // ------------------------------------------------------------------
    std::cout << "first render, to warm-up and compute reference image ...." << std::endl;
    renderFrame(compositor,replay,finalPixels);
    uint32_t *h_finalPixels = new uint32_t[fbSize.x*fbSize.y];
    cudaMemcpy(h_finalPixels,finalPixels,fbSize.x*fbSize.y*sizeof(uint32_t),cudaMemcpyDefault);
    if (rank == 0) {
      const std::string fileName = "bench_before.png";
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     h_finalPixels,fbSize.x*sizeof(uint32_t));
    }

    // ------------------------------------------------------------------
    // render N frames, and measure
    // ------------------------------------------------------------------
    double t0 = getCurrentTime();
    int numFrames = 200;
    for (int frameID=0;frameID<numFrames;frameID++) {
      renderFrame(compositor,replay,finalPixels);
    }
    double t1 = getCurrentTime();
    std::cout << "composited " << numFrames << " frames of "
              << fbSize.x << " x " << fbSize.y << " pixels " << " in "
              << (t1-t0) << " seconds" << std::endl;
    

    // ------------------------------------------------------------------
    // render one frame
    // ------------------------------------------------------------------
    std::cout << "dumping final reference image, to triple check ...." << std::endl;
    if (rank == 0) {
      const std::string fileName = "bench_after.png";
      cudaMemcpy(h_finalPixels,finalPixels,fbSize.x*fbSize.y*sizeof(uint32_t),cudaMemcpyDefault);
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     h_finalPixels,fbSize.x*sizeof(uint32_t));
    }

    PING;
    
    // ------------------------------------------------------------------
    // done bench
    // ------------------------------------------------------------------
    MPI_Finalize();
  }
}

int main(int ac, char **av)
{ dc::runBench(ac,av); return 0; }

