#include <cmath>
#include <random>
#include <chrono>
#include <array>
#include <iostream>
#define volume(V, Np, Ni) (V * (Np * (1.0 / Ni)))
typedef struct _float3{
    float x, y, z;
} float3;


float3 cross(const float3 &a, const float3 &b)

{
    float3 result{a.y * b.z - a.z * b.y,
                                  a.z * b.x - a.x * b.z,
                                  a.x * b.y - a.y * b.x};
    return result;
}
float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
float3 normalized(const float3 &a)
{
    float r = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    float3 norm_vector{a.x / r, a.y / r, a.z / r};
    return norm_vector;
}

float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

float3 operator-(float3 a, float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
} 

float3 operator*(float a, float3 b)
{
    return {a * b.x, a * b.y, a * b.z};
}

float3 operator/(float3 a, float b)
{
    return {a.x / b, a.y / b, a.z / b};
}
bool hasSameSign(float x, float y)
{
    union
    {
        float d;
        unsigned long long u;
    } ux = {x}, uy = {y};
    return (ux.u >> ((sizeof(float) * 8) - 1)) == (uy.u >> ((sizeof(float) * 8) - 1));
}

float weight_calculator(const float3 &P,
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

bool inside_tetrahedron(const float3 &A,
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

float volume_tetrahedron(const float3 &A,
                                  const float3 &B,
                                  const float3 &C,
                                  const float3 &D,
                                  int N)
{
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distributionX(-1.5, 0.3),
        distributionY(-0.2, 0.4),
        distributionZ(-0.7, 0.5);

    int count = 0;
    for (int i = 0; i < N; i++)
    {
        float3 P{distributionX(generator),
                 distributionY(generator),
                 distributionZ(generator)};

        if (inside_tetrahedron(A, B, C, D, P))
        {
            count++;
        }
    }
    return count;
}


int main(){
    std::array<float3, 4> vertices = {
        float3{0, 0, -0.7},   // A
        float3{-1.5, 0, 0},   // B
        float3{0, -0.2, 0},   // C
        float3{0.3, 0.4, 0.5} // D
    };
    float vol = (1.8 * 0.6 * 1.2);

    int N = 1'000'000;
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    float HOSTvolume = volume_tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3], N);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Volume of tetrahedron: " << volume(vol, HOSTvolume, N) << std::endl;
    std::cout << "Time: " << time_span.count() << " ms" << std::endl;
}