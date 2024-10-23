#include <pc/transpose.hpp>

void pc::matrix_transpose(int **matrix, int **out, std::size_t N, std::size_t M)
{
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = i; j < M; j++)
        {
            out[i][j] = matrix[j][i];
            out[j][i] = matrix[i][j];
        }
    }
    return;
}
