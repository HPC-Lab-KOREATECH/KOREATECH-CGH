// ======================================================================== //
// Copyright 2018-2019 Ingo Wald
//
// Licensed under the Apache License, Version 2.0
// ======================================================================== //

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    struct TriangleMeshSBTData {
        vec3f color;
        vec3f* vertex;
        vec3f* normal;
        vec2f* texcoord;
        vec3i* index;
        bool hasTexture;
        cudaTextureObject_t texture;
    };

    struct LaunchParams
    {
        struct {
            float* color_buffer_r;
            float* color_buffer_g;
            float* color_buffer_b;
            float* depth_buffer;
            float* z_buffer;
            vec2i size;
        } frame;

        struct {
            vec3f position;
            vec3f direction;
            vec3f horizontal;
            vec3f vertical;
        } camera;

        OptixTraversableHandle traversable;
    };

} // ::osc
