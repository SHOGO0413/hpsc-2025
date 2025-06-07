// ... (前回の修正後のコードの続き)

    if (rank == 0) {
        ufile.close();
        vfile.close();
        pfile.close();
    } // この閉じ波括弧が不足していたか、行われたはず。

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0)
    { // ランク0のみが出力
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0;
} // main関数の閉じ波括弧
