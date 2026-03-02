import matplotlib.pyplot as plt
import numpy as np
import imageio


def render_epicycle_gif(image_np, objects, save_path,
                         frames=200, K=50):

    H, W, _ = image_np.shape
    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
    trails = [[] for _ in objects]
    frames_out = []

    for f in range(frames):
        t = f / frames

        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        ax.imshow(image_np, alpha=0.45)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # image coordinate
        ax.axis("off")

        for i, (coeffs, freqs, center) in enumerate(objects):
            pos = draw_epicycles(
                ax, coeffs, freqs, center, t, K, colors[i]
            )

            trails[i].append(pos)
            trail = np.array(trails[i])
            ax.plot(trail.real, trail.imag, color=colors[i], linewidth=2)

            ax.scatter(pos.real, pos.imag, color=colors[i], s=15)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames_out.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames_out, fps=20)
    print(f"Saved epicycle GIF: {save_path}")

def draw_epicycles(ax, coeffs, freqs, center, t, K, color):
    pos = center[0] + 1j * center[1]

    for i in range(K):
        prev = pos
        radius = np.abs(coeffs[i])
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