import matplotlib.pyplot as plt
import numpy as np
import os

NX = 41
NY = 41

def main():
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    with open('u.dat', 'r') as f:
        uraw = f.readlines()
    with open('v.dat', 'r') as f:
        vraw = f.readlines()
    with open('p.dat', 'r') as f:
        praw = f.readlines()

    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True) # 既に存在してもエラーにならない

    for n in range(len(uraw)):
        plt.clf()

        u_flattened = [float(val) for val in uraw[n].strip().split() if val]
        v_flattened = [float(val) for val in vraw[n].strip().split() if val]
        p_flattened = [float(val) for val in praw[n].strip().split() if val]

        expected_len_per_line = NX * NY
        if len(u_flattened) != expected_len_per_line or \
           len(v_flattened) != expected_len_per_line or \
           len(p_flattened) != expected_len_per_line:
            print(f"警告: タイムステップ {n} でデータ長が期待値と異なります！")

        for j in range(NY):
            for i in range(NX):
                idx = j * NX + i
                u[j, i] = u_flattened[idx]
                v[j, i] = v_flattened[idx]
                p[j, i] = p_flattened[idx]

        plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        plt.title(f'C++, n = {n}')

        frame_filename = os.path.join(output_dir, f'cavity_frame_{n:04d}.png')
        plt.savefig(frame_filename)
        print(f"Saved {frame_filename}")

    plt.close()）

if __name__ == '__main__':
    main()
