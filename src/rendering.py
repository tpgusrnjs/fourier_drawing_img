import matplotlib.pyplot as plt
import numpy as np
import imageio

from utils.signal import epicycle_position

def render_epicycle_gif(objects, save_path,
                         frames=200, K=50,
                         canvas_size=800):

    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
    trails = [[] for _ in objects]
    gif_frames = []

    for f in range(frames):
        t = f / frames
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_aspect("equal")
        ax.axis("off")

        for i, (coeffs, freqs) in enumerate(objects):
            pos = draw_epicycles(
                ax,
                coeffs,
                freqs,
                t,
                K=40,
                color=colors[i]
            )

            trails[i].append(pos)
            trail = np.array(trails[i])
            ax.plot(trail.real, trail.imag, color=colors[i], linewidth=2)

            ax.scatter(pos.real, pos.imag, color=colors[i], s=20)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        gif_frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, gif_frames, fps=20)
    print(f"Saved epicycle GIF: {save_path}")

def draw_epicycles(ax, coeffs, freqs, t, K, color):
    pos = 0j

    for i in range(K):
        prev = pos
        radius = np.abs(coeffs[i])
        phase = np.angle(coeffs[i])
        k = freqs[i]

        pos += coeffs[i] * np.exp(2j * np.pi * k * t)

        # draw circle
        circle = plt.Circle(
            (prev.real, prev.imag),
            radius,
            color=color,
            fill=False,
            alpha=0.3,
            linewidth=1
        )
        ax.add_patch(circle)

        # draw arm
        ax.plot(
            [prev.real, pos.real],
            [prev.imag, pos.imag],
            color=color,
            linewidth=1
        )

    return pos