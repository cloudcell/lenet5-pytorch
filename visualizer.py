import matplotlib.pyplot as plt
import numpy as np


class PredVisualizer(object):
    """A simple prediction dynamics visualizer for classifiers."""
    def __init__(self, batch, save_dir):
        self.save_dir = save_dir
        self.batch = batch  # (X, y) Torch tensor
        self.probs = []  # probabilities

    def add_preds(self, probs, step):
        self.probs.append(probs)
        self.save_vis(step)

    def save_vis(self, step):
        p = self.probs[-1]
        N, C = p.shape
        H = int(np.sqrt(N))

        classes = range(C)
        for i in range(H):
            for j in range(H):
                idx = i * H + j
                plt.subplot(H, H, idx + 1)
                plt.bar(classes, p[idx, :].numpy())
                gt_label = str(self.batch[1].numpy()[idx])
                plt.title('GT = ' + gt_label)
                plt.xticks(classes, fontsize=8)
                plt.yticks(fontsize=8)
                plt.ylim((0., 1.))
        plt.tight_layout()

        plt.savefig(self.save_dir + '/vis_' + '{:05d}'.format(step) + '.png') #, dpi=100)
        plt.close()
