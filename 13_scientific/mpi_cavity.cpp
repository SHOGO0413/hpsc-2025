#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

// 行列の型定義
typedef vector<vector<float>> matrix;

int main(int argc, char *argv[])
{
    // MPIの初期化
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 総プロセス数の取得
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 現在のプロセスのランク（ID）の取得

    // シミュレーションパラメータ
    int nx = 41; // x方向のグリッド点数
    int ny = 41; // y方向のグリッド点数
    int nt = 500; // 時間ステップの総数
    int nit = 50; // 圧力ポアソン方程式の反復回数
    double dx = 2. / (nx - 1); // x方向のグリッド間隔
    double dy = 2. / (ny - 1); // y方向のグリッド間隔
    double dt = .01; // 時間ステップの大きさ
    double rho = 1.; // 流体の密度
    double nu = .02; // 流体の動粘性係数

    // 各プロセスの担当するx方向のグローバルインデックス範囲を計算
    int local_nx_base = nx / size;     // 各プロセスに割り当てる基本的なグリッド数
    int remainder = nx % size;         // プロセス数で割り切れない部分

    int begin_idx; // 各プロセスが担当するx方向の開始インデックス (グローバルインデックス)
    int end_idx;   // 各プロセスが担当するx方向の終了インデックス (グローバルインデックス, 排他的)

    if (rank < remainder)
    { // 余り分を処理するプロセス（多めに担当）
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
    }
    else
    { // 余りを考慮しないプロセス（基本数を担当）
        begin_idx = rank * local_nx_base + remainder;
        end_idx = begin_idx + local_nx_base;
    }

    // ゴーストセルを含まない、各プロセスが担当する列の数
    int my_cols = end_idx - begin_idx;

    // グローバル配列 (全てのプロセスがnx*nyのメモリを持つ)
    // ただし、計算は各プロセスの担当範囲で行う
    matrix u(ny, vector<float>(nx));
    matrix v(ny, vector<float>(nx));
    matrix p(ny, vector<float>(nx));
    matrix b(ny, vector<float>(nx));
    matrix un(ny, vector<float>(nx)); // 前の時間ステップの u
    matrix vn(ny, vector<float>(nx)); // 前の時間ステップの v
    matrix pn(ny, vector<float>(nx)); // 前の反復ステップの p

    // --- 初期化ループ ---
    // 各プロセスは自分の担当範囲のみを初期化
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
    
    // 時間計測開始
    auto start = high_resolution_clock::now();

    // 出力ファイルストリーム (ランク0のみが使用する想定だが、ここではすべてのプロセスで開いておく)
    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    // バッファの用意: 1列分のデータを送受信するためのテンポラリ配列
    vector<float> send_left_buffer(ny);  // 左隣に送るデータ
    vector<float> recv_left_buffer(ny);  // 左隣から受け取るデータ
    vector<float> send_right_buffer(ny); // 右隣に送るデータ
    vector<float> recv_right_buffer(ny); // 右隣から受け取るデータ

    // MPI_Gatherv に必要な変数
    // 各プロセスが送信するデータ数を計算し、ルートプロセス（ランク0）で受信するために使用
    vector<int> gatherv_recvcounts(size); // 各プロセスから受け取るデータの総要素数 (列数 * ny)
    vector<int> gatherv_displs(size);     // 各プロセスからのデータを受信バッファのどこに配置するか (オフセット)

    for (int i = 0; i < size; ++i)
    {
        int current_rank_cols;
        if (i < remainder)
        {
            current_rank_cols = local_nx_base + 1;
        }
        else
        {
            current_rank_cols = local_nx_base;
        }
        gatherv_recvcounts[i] = current_rank_cols * ny;
    }

    gatherv_displs[0] = 0;
    for (int i = 1; i < size; ++i)
    {
        gatherv_displs[i] = gatherv_displs[i - 1] + gatherv_recvcounts[i - 1];
    }

    // ランク0で全体の計算結果を受け取るためのバッファ
    // 1次元配列として確保し、[j * nx + i] でアクセスする
    vector<float> global_u(nx * ny);
    vector<float> global_v(nx * ny);
    vector<float> global_p(nx * ny);

    // --- 初期状態の同期 (最初のゴーストセル交換で0を送り合わないようにするため) ---
    // 各プロセスが担当範囲のデータを初期化した後、MPI_Allgathervを使って全プロセスで配列を同期する
    // これにより、時間ステップ1のゴーストセル交換の時点で、すべての要素が正しい初期値（0）を持つ
    vector<float> initial_send_u_flat(my_cols * ny);
    vector<float> initial_send_v_flat(my_cols * ny);
    vector<float> initial_send_p_flat(my_cols * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i_local = 0; i_local < my_cols; ++i_local) {
            // ここが修正箇所: u[j][begin_idx + i_local] は正しいアクセス
            initial_send_u_flat[j * my_cols + i_local] = u[j][begin_idx + i_local]; 
            initial_send_v_flat[j * my_cols + i_local] = v[j][begin_idx + i_local];
            initial_send_p_flat[j * my_cols + i_local] = p[j][begin_idx + i_local];
        }
    }
    
    // MPI_Allgathervで全てのプロセスのu, v, p配列を同期する
    // 受信側はu[0].data() (matrix u(ny, vector<float>(nx))の先頭)
    MPI_Allgatherv(initial_send_u_flat.data(), gatherv_recvcounts[rank], MPI_FLOAT, 
                   u[0].data(), gatherv_recvcounts.data(), gatherv_displs.data(), MPI_FLOAT, 
                   MPI_COMM_WORLD);
    MPI_Allgatherv(initial_send_v_flat.data(), gatherv_recvcounts[rank], MPI_FLOAT, 
                   v[0].data(), gatherv_recvcounts.data(), gatherv_displs.data(), MPI_FLOAT, 
                   MPI_COMM_WORLD);
    MPI_Allgatherv(initial_send_p_flat.data(), gatherv_recvcounts[rank], MPI_FLOAT, 
                   p[0].data(), gatherv_recvcounts.data(), gatherv_displs.data(), MPI_FLOAT, 
                   MPI_COMM_WORLD);
    // --- 初期状態の同期 終わり ---


    // --- メインの時間ステップループ ---
    for (int n = 0; n < nt; n++)
    {
        // --- 速度場 (u, v) のゴーストセル交換 ---
        // b, u, v の計算の前に、隣接プロセスから u, v の境界データを取得する
        // MPI_Sendrecv を使用して、データの送受信を同時に行う
        // 各行(j)ごとに通信が必要なため、ny回の通信 (より効率的な方法はMPI_Type_vector)

        // u のゴーストセル交換 (タグ 0,1)
        if (rank > 0) // 左隣のプロセスがいる場合
        { 
            for (int j = 0; j < ny; j++) send_left_buffer[j] = u[j][begin_idx]; // 自分の左端の列
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0, // 送信
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1, // 受信
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) u[j][begin_idx - 1] = recv_left_buffer[j]; // 左側のゴーストセルを更新
        }
        if (rank < size - 1) // 右隣のプロセスがいる場合
        { 
            for (int j = 0; j < ny; j++) send_right_buffer[j] = u[j][end_idx - 1]; // 自分の右端の列
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1, // 送信
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0, // 受信
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) u[j][end_idx] = recv_right_buffer[j]; // 右側のゴーストセルを更新
        }
        
        // v のゴーストセル交換 (u と同様のロジック, タグ 2,3)
        if (rank > 0)
        {
            for (int j = 0; j < ny; j++) send_left_buffer[j] = v[j][begin_idx];
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 2,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) v[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1)
        {
            for (int j = 0; j < ny; j++) send_right_buffer[j] = v[j][end_idx - 1];
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 3,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) v[j][end_idx] = recv_right_buffer[j];
        }
        // --- 速度場 (u, v) のゴーストセル交換 終わり ---


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
            // --- 圧力場 (p/pn) のゴーストセル交換 ---
            // p の計算の前に、隣接プロセスから p の境界データを取得する
            // pn にコピーした p の値を使うため、pnのゴーストセルを更新する
            // (pnはpの古い値なので、pの現在の値でゴーストセルを更新してからpnにコピーする)
            if (rank > 0)
            {
                for (int j = 0; j < ny; j++) send_left_buffer[j] = p[j][begin_idx];
                MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 4,
                             &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 5,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j) p[j][begin_idx - 1] = recv_left_buffer[j];
            }
            if (rank < size - 1)
            {
                for (int j = 0; j < ny; j++) send_right_buffer[j] = p[j][end_idx - 1];
                MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 5,
                             &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 4,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j) p[j][end_idx] = recv_right_buffer[j];
            }
            // --- 圧力場 (p/pn) のゴーストセル交換 終わり ---

            for (int j = 0; j < ny; j++)
            {
                for (int i = begin_idx; i < end_idx; i++)
                {
                    pn[j][i] = p[j][i];
                }
            }
            // --- 追加: pn のゴーストセル交換 ---
            // pn は p のコピーであり、p のゴーストセルは更新済みだが、pn自身のゴーストセルは未更新
            // p の計算で pn のゴーストセルが必要なため、pn のゴーストセルを更新する
            if (rank > 0)
            {
                for (int j = 0; j < ny; j++) send_left_buffer[j] = pn[j][begin_idx];
                MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 6, // 新しいタグを使用
                             &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 7, // 新しいタグを使用
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j) pn[j][begin_idx - 1] = recv_left_buffer[j];
            }
            if (rank < size - 1)
            {
                for (int j = 0; j < ny; j++) send_right_buffer[j] = pn[j][end_idx - 1];
                MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 7, // 新しいタグを使用
                             &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 6, // 新しいタグを使用
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j) pn[j][end_idx] = recv_right_buffer[j];
            }
            // --- pn のゴーストセル交換 終わり ---


            for (int j = 1; j < ny - 1; j++)
            {
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

        // un, vn へのコピーを nit ループの外に移動
        for (int j = 0; j < ny; j++)
        {
            for (int i = begin_idx; i < end_idx; i++)
            {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        // --- 追加: un と vn のゴーストセル交換 ---
        // un と vn は u, v の計算で参照されるため、その前に境界情報を同期する
        // un のゴーストセル交換 (タグ 8,9)
        if (rank > 0) 
        { 
            for (int j = 0; j < ny; j++) send_left_buffer[j] = un[j][begin_idx]; 
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 8, 
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 9, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) un[j][begin_idx - 1] = recv_left_buffer[j]; 
        }
        if (rank < size - 1) 
        { 
            for (int j = 0; j < ny; j++) send_right_buffer[j] = un[j][end_idx - 1]; 
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 9, 
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 8, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) un[j][end_idx] = recv_right_buffer[j]; 
        }
        
        // vn のゴーストセル交換 (タグ 10,11)
        if (rank > 0) 
        {
            for (int j = 0; j < ny; j++) send_left_buffer[j] = vn[j][begin_idx];
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 10,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) vn[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1)
        {
            for (int j = 0; j < ny; j++) send_right_buffer[j] = vn[j][end_idx - 1];
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 11,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j) vn[j][end_idx] = recv_right_buffer[j];
        }
        // --- un と vn のゴーストセル交換 終わり ---


        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                // Compute u[j][i] and v[j][i]
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2. * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / (dx * dx) * (un[j][i + 1] - 2. * un[j][i] + un[j][i - 1]) + nu * dt / (dy * dy) * (un[j + 1][i] - 2. * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un
