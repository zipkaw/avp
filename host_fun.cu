#include <random>
#include <chrono>
#include <array>

#include "vect_fun.cu"

__host__ float volume_tetrahedron(const float3 &A,
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
