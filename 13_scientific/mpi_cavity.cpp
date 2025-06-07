#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

typedef vector<vector<float>> matrix;

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    int local_nx_base = nx / size; // 各プロセス用の割り当てグリッド数
    int remainder = nx % size;      // プロセス数で割り切れない部分。（41%4=1）

    int begin_idx;
    int end_idx;

    if (rank < remainder)
    { // 余り分を処理するプロセス。
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
    }
    else
    { // 余りを考慮しないプロセス。
        begin_idx = rank * local_nx_base + remainder;
        end_idx = begin_idx + local_nx_base;
    }

    matrix u(ny, vector<float>(nx));
    matrix v(ny, vector<float>(nx));
    matrix p(ny, vector<float>(nx));
    matrix b(ny, vector<float>(nx));
    matrix un(ny, vector<float>(nx));
    matrix vn(ny, vector<float>(nx));
    matrix pn(ny, vector<float>(nx));

    for (int j = 0; j < ny; j++)
    {
        for (int i = begin_idx; i < end_idx; i++)
        {
            u[j][i] = 0;
            v[j][i] = 0;
            p[j][i] = 0;
            b[j][i] = 0;
        }
    }
    auto start = high_resolution_clock::now();

    ofstream ufile, vfile, pfile;
    if (rank == 0) {
        ufile.open("u.dat");
        vfile.open("v.dat");
        pfile.open("p.dat");
    }

    // ここでsend_count, request, statusを定義する
    // これにより、ntループ内で複数回再定義されなくなり、かつPとU/Vの境界データ交換で共有できるようになる
    MPI_Request request[4];
    MPI_Status status[4];
    int send_count = ny; // 1列分のデータ数 (これはnyに依存するので、ntループの外で定義可能)

    for (int n = 0; n < nt; n++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                // Compute b[j][i]
                b[j][i] = rho * (1. / dt * ((u[j][i + 1] - u[j][i - 1]) / (2. * dx) + (v[j + 1][i] - v[j - 1][i]) / (2. * dy)) -
                                 pow((u[j][i + 1] - u[j][i - 1]) / (2. * dx), 2) -
                                 2. * ((u[j + 1][i] - u[j - 1][i]) / (2. * dy) * (v[j][i + 1] - v[j][i - 1]) / (2. * dx)) -
                                 pow((v[j + 1][i] - v[j - 1][i]) / (2. * dy), 2));
            }
        }

        for (int it = 0; it < nit; it++)
        {
            // pnへのコピー
            for (int j = 0; j < ny; j++)
            {
                for (int i = begin_idx; i < end_idx; i++)
                {
                    pn[j][i] = p[j][i];
                }
            }

            // --- Pの境界データ交換（ゴーストセルの更新） ---
            // 各プロセスは、自分の担当範囲の境界（1列）を隣接プロセスに送り、
            // 同時に隣接プロセスから境界データ（ゴーストセル）を受け取る
            // ここでのrequest, status, send_countの定義は削除され、外部の定義を使用

            // 左隣のプロセスとデータを交換 (begin_idx - 1 のゴーストセルを更新)
            if (rank > 0)
            {
                // 左に自分の左端のデータ (begin_idx) を送信
                MPI_Isend(&pn[0][begin_idx], send_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
                // 左から右隣のデータ (begin_idx - 1) を受信
                MPI_Irecv(&pn[0][begin_idx - 1], send_count, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &request[1]);
            }

            // 右隣のプロセスとデータを交換 (end_idx のゴーストセルを更新)
            if (rank < size - 1)
            {
                // 右に自分の右端のデータ (end_idx - 1) を送信
                MPI_Isend(&pn[0][end_idx - 1], send_count, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &request[2]);
                // 右から左隣のデータ (end_idx) を受信
                MPI_Irecv(&pn[0][end_idx], send_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &request[3]);
            }

            // 非同期通信の完了を待つ
            if (rank > 0) MPI_Waitall(2, request, status);
            if (rank < size - 1) MPI_Waitall(2, request + 2, status + 2);
            // --- Pの境界データ交換終了 ---

            for (int j = 1; j < ny - 1; j++)
            {
                // begin_idxが0でない場合、begin_idx-1はゴーストセル
                // end_idxがnx-1でない場合、end_idxはゴーストセル
                // pn[j][i+1] と pn[j][i-1] のアクセスが、begin_idx-1 や end_idx に
                // なる可能性があるため、上記の通信が必要。
                for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
                {
                    // Compute p[j][i]
                    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                               dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                               b[j][i] * dx * dx * dy * dy) /
                              (2. * (dx * dx + dy * dy));
                }
            }
            if (rank == 0)
            { // ランク0が左端の境界条件を担当
                for (int j = 0; j < ny; j++)
                {
                    p[j][0] = p[j][1];
                }
            }
            if (rank == size - 1)
            { // 最後のランクが右端の境界条件を担当
                for (int j = 0; j < ny; j++)
                {
                    p[j][nx - 1] = p[j][nx - 2];
                }
            }
            for (int i = begin_idx; i < end_idx; i++)
            { // y方向の境界条件は共有メモリ上でそのまま
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0;
            }
        } // end of nit loop

        // un, vn へのコピー
        for (int j = 0; j < ny; j++)
        {
            for (int i = begin_idx; i < end_idx; i++)
            {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        // --- U, Vの境界データ交換（ゴーストセルの更新） ---
        // Pと同様に、UとVも隣接プロセスとの間で境界データを交換する
        // 速度場の計算前に最新の隣接データが必要
        // U
        if (rank > 0)
        {
            MPI_Isend(&un[0][begin_idx], send_count, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&un[0][begin_idx - 1], send_count, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD, &request[1]);
        }
        if (rank < size - 1)
        {
            MPI_Isend(&un[0][end_idx - 1], send_count, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(&un[0][end_idx], send_count, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, &request[3]);
        }
        if (rank > 0) MPI_Waitall(2, request, status);
        if (rank < size - 1) MPI_Waitall(2, request + 2, status + 2);

        // V
        if (rank > 0)
        {
            MPI_Isend(&vn[0][begin_idx], send_count, MPI_FLOAT, rank - 1, 4, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&vn[0][begin_idx - 1], send_count, MPI_FLOAT, rank - 1, 5, MPI_COMM_WORLD, &request[1]);
        }
        if (rank < size - 1)
        {
            MPI_Isend(&vn[0][end_idx - 1], send_count, MPI_FLOAT, rank + 1, 5, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(&vn[0][end_idx], send_count, MPI_FLOAT, rank + 1, 4, MPI_COMM_WORLD, &request[3]);
        }
        if (rank > 0) MPI_Waitall(2, request, status);
        if (rank < size - 1) MPI_Waitall(2, request + 2, status + 2);
        // --- U, Vの境界データ交換終了 ---


        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                // Compute u[j][i] and v[j][i]
                // un[j][i+1], un[j][i-1], vn[j][i+1], vn[j][i-1] のアクセスでゴーストセルが必要
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2. * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / (dx * dx) * (un[j][i + 1] - 2. * un[j][i] + un[j][i - 1]) + nu * dt / (dy * dy) * (un[j + 1][i] - 2. * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2. * rho * dy) * (p[j + 1][i] - p[j - 1][i]) // ここはdyに修正
                                 + nu * dt / (dx * dx) * (vn[j][i + 1] - 2. * vn[j][i] + vn[j][i - 1]) + nu * dt / (dy * dy) * (vn[j + 1][i] - 2. * vn[j][i] + vn[j - 1][i]);
            }
        }
        if (rank == 0)
        { // ランク0が左端の境界条件を担当
            for (int j = 0; j < ny; j++)
            {
                u[j][0] = 0;
                v[j][0] = 0;
            }
        }
        if (rank == size - 1)
        { // 最後のランクが右端の境界条件を担当
            for (int j = 0; j < ny; j++)
            {
                u[j][nx - 1] = 0;
                v[j][nx - 1] = 0;
            }
        }
        for (int i = begin_idx; i < end_idx; i++)
        { // y方向の境界条件は共有メモリ上でそのまま
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }

        if (n % 10 == 0)
        {
            if (rank == 0)
