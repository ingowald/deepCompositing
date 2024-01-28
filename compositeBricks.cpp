// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <experimental/filesystem>

#include "owl/common/math/vec.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

using namespace owl::common;

inline uint32_t make_8bit(const float f)
{
  return std::min(255,std::max(0,int(f*256.f)));
}

inline uint32_t make_rgba(const vec3f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (0xffU << 24);
}
inline uint32_t make_rgba(const vec4f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (make_8bit(color.w) << 24);
}

vec3i guessBrickID(std::string fileName)
{
  vec3i result(-1);
  int x=0,y=0,z=0;
  if (sscanf(fileName.c_str(),"sub-dns-%i-%i-%i.rgbaz",&x,&y,&z) == 3)
    result = {x,y,z};
  return result;
}

bool isMine(vec3i idx,
            int mpiRank,
            int mpiSize)
{
  if (mpiSize == 1) return true;

  if (mpiSize == 2) {
    if (mpiRank == 0 && idx.y < 5 || mpiRank == 1 && idx.y >= 5) return true;

    //if ((mpiRank == 0 && (idx.x % 2) == 0) || (mpiRank == 1 && (idx.x % 2) == 1)) return true;
  }

  return false;
}

struct Fragment { float a,r,g,b,z; };

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: compositeBricks <imgDir> [<mpiSize>]\n";
    return EXIT_FAILURE;
  }

  namespace fs = std::experimental::filesystem;

  std::string imgDir(argv[1]);
  int mpiSize = argc == 3 ? std::stoi(argv[2]) : 1;

  constexpr unsigned MaxFragments = 1<<5;
  typedef std::array<Fragment,MaxFragments> Fragments;

  vec2i fbSize = { -1,-1 };


  for (int rank=0; rank<mpiSize; ++rank) {
    // Assemble, for each pixel, an unordered list
    // of fragments from bricks on this rank
    std::vector<Fragments> fragments;
    int bricksLoaded = 0;
    for (auto &dirEntry: fs::directory_iterator(imgDir)) {
      fs::path p = dirEntry.path();
      if (p.extension() == ".rgbaz") {
        if (isMine(guessBrickID(p.filename().string()),rank,mpiSize)) {
          bricksLoaded++;
          std::ifstream in(p.c_str(),std::ios::binary);
          in.read((char*)&fbSize,sizeof(fbSize));
          std::vector<uint32_t> fbRGBA(fbSize.x*fbSize.y);
          std::vector<float> fbDepth(fbSize.x*fbSize.y);
          in.read((char*)fbRGBA.data(),fbSize.x*fbSize.y*sizeof(uint32_t));
          in.read((char*)fbDepth.data(),fbSize.x*fbSize.y*sizeof(float));
  
          if (fragments.empty()) {
            fragments.resize(fbSize.x*fbSize.y);
            for (auto &frg : fragments)
              std::fill(frg.begin(),frg.end(),(Fragment){0.f,0.f,0.f,0.f,1e20f});
          }

          for (int y=0; y<fbSize.y; ++y) {
            for (int x=0; x<fbSize.x; ++x) {
              if (fbDepth[y*fbSize.x+x] == 1e10f)
                continue;

              uint32_t rgba = fbRGBA[y*fbSize.x+x];
              vec4f color((rgba & 0xff) / 255.f,
                          ((rgba >> 8) & 0xff) / 255.f,
                          ((rgba >> 16) & 0xff) / 255.f,
                          ((rgba >> 24) & 0xff) / 255.f);
              for (unsigned i=0; i<MaxFragments; ++i)
              {
                if (fragments[y*fbSize.x+x][i].z == 1e20f) { // free slot in frg list
                  Fragment &f = fragments[y*fbSize.x+x][i];
                  f.z = fbDepth[y*fbSize.x+x];
                  f.a = color.w;
                  f.r = color.x;
                  f.g = color.y;
                  f.b = color.z;
                  break;
                }
              }
            }
          }
        }
      }
    }

    std::cout << "Done reading, loaded " << bricksLoaded << " bricks...\n";


    // Sort those list into visibility order
    for (int y=0; y<fbSize.y; ++y) {
      for (int x=0; x<fbSize.x; ++x) {
        std::sort(fragments[y*fbSize.x+x].begin(),fragments[y*fbSize.x+x].end(),
                  [](Fragment a, Fragment b) { return a.z < b.z; });
      }
    }

    std::cout << "Done sorting...\n";


    // Local compositing to intermediate image
    std::vector<float> fbDepth(fbSize.x*fbSize.y);
    std::vector<uint32_t> fbRGBA(fbSize.x*fbSize.y);

    for (int y=0; y<fbSize.y; ++y) {
      for (int x=0; x<fbSize.x; ++x) {
        const Fragments &frg = fragments[y*fbSize.x+x];
        float alpha = 0.f;
        vec3f color = 0.f;
        for (unsigned i=0; i<MaxFragments; ++i) {
          if (frg[i].z == 1e20f) {
            if (i==0) { // At least one (far away) fragment
              fbRGBA[y*fbSize.x+x] = 0;
              fbDepth[y*fbSize.x+x] = 1e10f;
            }
            break;
          }

          color = color
            +  (1.f-alpha)
            *  vec3f((float)frg[i].r,
                     (float)frg[i].g,
                     (float)frg[i].b);
          alpha += (1.f-alpha)*float(frg[i].a);
        }
        fbRGBA[y*fbSize.x+x] = make_rgba(vec4f(color,alpha));
        fbDepth[y*fbSize.x+x] = fragments[y*fbSize.x+x][0].z; // depth of 1st fragment!
        //if (alpha > 0.f)
        //  std::cout << x << ' ' << y << ": " << vec4f(color,alpha) << ' ' << fragments[y*fbSize.x+x][0].z << ' ' << fbDepth[y*fbSize.x+x] << '\n';
      }
    }

    std::cout << "Done compositing...\n";


    // Write out to file

    std::stringstream ss;
    ss << "sub-dns-" << rank << ".rgbaz";
    std::ofstream os(ss.str().c_str(),std::ios::binary);
    os.write((const char *)&fbSize,sizeof(fbSize));
    os.write((const char *)fbRGBA.data(),fbRGBA.size()*sizeof(uint32_t));
    os.write((const char *)fbDepth.data(),fbDepth.size()*sizeof(float));

#if 1
    ss << ".png";
    stbi_write_png(ss.str().c_str(),fbSize.x,fbSize.y,4,
                   fbRGBA.data(),fbSize.x*sizeof(uint32_t));
#endif

    std::cout << "Done on rank << " << rank << " of " << mpiSize << "...\n";
  }
  std::cout << "Done!\n";
}
