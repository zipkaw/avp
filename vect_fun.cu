
#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__host__ __device__ float3 cross(const float3 &a, const float3 &b)

{
    float3 result = make_float3(a.y * b.z - a.z * b.y,
                                  a.z * b.x - a.x * b.z,
                                  a.x * b.y - a.y * b.x);
    return result;
}
__host__ __device__ float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ float3 normalized(const float3 &a)
{
    float r = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    float3 norm_vector = make_float3(a.x / r, a.y / r, a.z / r);
    return norm_vector;
}

__host__ __device__ float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ float3 operator-(float3 a, float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ float3 operator*(float a, float3 b)
{
    return {a * b.x, a * b.y, a * b.z};
}

__host__ __device__ float3 operator/(float3 a, float b)
{
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ bool hasSameSign(float x, float y)
{
    union
    {
        float d;
        unsigned long long u;
    } ux = {x}, uy = {y};
    return (ux.u >> ((sizeof(float) * 8) - 1)) == (uy.u >> ((sizeof(float) * 8) - 1));
}

__host__ __device__ float weight_calculator(const float3 &P,
                                            const float3 &B,
                                            const float3 &C,
                                            const float3 &D,
                                            const float3 &A)
{
    /*
              P is point to face {BCD}
    weight = -------------------------
              A is point to face {BCD}
    */
    float3 normal = cross(C - B, D - B);
    float distanceP = dot(normal, P - B);
    float distanceA = dot(normal, A - B);

    return distanceP/distanceA;

}

__host__ __device__ bool inside_tetrahedron(const float3 &A,
                                            const float3 &B,
                                            const float3 &C,
                                            const float3 &D,
                                            const float3 &P)
{
    
    float alfa = weight_calculator(P, B, C, D, A);
    float beta = weight_calculator(P, A, C, D, B);
    float gamma = weight_calculator(P, A, B, D, C);
    float delta = weight_calculator(P, A, B, C, D);

    return alfa >= 0 &&
           beta >= 0 &&
           gamma >= 0 &&
           delta >= 0;
}