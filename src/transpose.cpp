#include <pc/transpose.hpp>

void pc::matTranspose(float **M, float **T, std::size_t N)
{
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = i; j < N; j++)
        {
            T[i][j] = M[j][i];
            T[j][i] = M[i][j];
        }
    }
    return;
}

bool pc::checkSym(float **M, std::size_t N)
{
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            if (M[i][j] != M[j][i])
            {
                return false;
            }
        }
    }
    return true;
}
