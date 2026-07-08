// ======================================================================== //
// Copyright 2018-2019 Ingo Wald
//
// Licensed under the Apache License, Version 2.0
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "helper_math.h"

using namespace osc;

struct pixel_value {
    gdt::vec3f albedo;
    float depth;
    float z_coord;
};

namespace osc {

    extern "C" __constant__ LaunchParams optixLaunchParams;

    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        float3 N;
        if (sbtData.normal) {
            N = (1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z];
        }
        else {
            const vec3f& A = sbtData.vertex[index.x];
            const vec3f& B = sbtData.vertex[index.y];
            const vec3f& C = sbtData.vertex[index.z];
            N = normalize(cross(B - A, C - A));
        }
        N = normalize(N);

        vec3f diffuseColor = sbtData.color;
        if (sbtData.hasTexture && sbtData.texcoord) {
            const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];

            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, -tc.y);
            diffuseColor *= (vec3f)fromTexture;
        }

        const float3 ray_dir = optixGetWorldRayDirection();
        const float cosDN = 0.2f + .8f * fabsf(dot(ray_dir, N));
        pixel_value& prd = *(pixel_value*)getPRD<pixel_value>();
        prd.albedo = cosDN * diffuseColor;
        const float3 P = optixGetWorldRayOrigin() + (optixGetRayTmax() * ray_dir);
        prd.depth = optixGetRayTmax();
        prd.z_coord = P.z;
    }

    extern "C" __global__ void __anyhit__radiance()
    {
    }

    extern "C" __global__ void __miss__radiance()
    {
        pixel_value& prd = *(pixel_value*)getPRD<pixel_value>();
        prd.albedo = vec3f(0.f);
        prd.depth = 0;
        prd.z_coord = 0;
    }

    extern "C" __global__ void __raygen__renderFrameOrthogonal()
    {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        pixel_value pixelColorPRD{};
        uint32_t u0, u1;
        packPointer(&pixelColorPRD, u0, u1);

        vec3f rayDir = normalize(vec3f(0, 0, optixLaunchParams.camera.direction.z));
        vec3f orth_camera_position(ix - optixLaunchParams.frame.size.x / 2 + 0.5f, iy - optixLaunchParams.frame.size.y / 2 + 0.5f, 0);

        optixTrace(optixLaunchParams.traversable,
            orth_camera_position,
            rayDir,
            0.f,
            1e20f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SURFACE_RAY_TYPE,
            RAY_TYPE_COUNT,
            SURFACE_RAY_TYPE,
            u0, u1);

        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.color_buffer_r[fbIndex] = pixelColorPRD.albedo.x;
        optixLaunchParams.frame.color_buffer_g[fbIndex] = pixelColorPRD.albedo.y;
        optixLaunchParams.frame.color_buffer_b[fbIndex] = pixelColorPRD.albedo.z;
        optixLaunchParams.frame.depth_buffer[fbIndex] = pixelColorPRD.depth;
        optixLaunchParams.frame.z_buffer[fbIndex] = pixelColorPRD.z_coord;
    }
} // ::osc
