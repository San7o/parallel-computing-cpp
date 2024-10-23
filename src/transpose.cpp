#include <pc/transpose.hpp>
#include <tenno/ranges.hpp>

void pc::matTranspose(float **M, float **T, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            T[i][j] = M[j][i];
            T[j][i] = M[i][j];
        }
    }
    return;
}

bool pc::checkSym(float **M, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            if (M[i][j] != M[j][i])
            {
                return false;
            }
        }
    }
    return true;
}
