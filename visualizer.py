import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Fashion-MNIST labels in order.
labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


class PredVisualizer(object):
    """A simple prediction dynamics visualizer for classifiers."""
    def __init__(self, batch, save_dir):
        self.save_dir = save_dir
        self.batch = batch  # (X, y) Torch tensor
        self.probs = []  # probabilities
        self.conf_mats = []

    def add_preds(self, probs, step):
        self.probs.append(probs)
        self._save_preds_vis(step)

    def add_conf_mat(self, cm, step):
        self.conf_mats.append(cm)
        self._save_conf_mat_vis(step)

    def _save_preds_vis(self, step):
        p = self.probs[-1]
        N, C = p.shape
        H = int(np.sqrt(N))

        classes = range(C)
        for i in range(H):
            for j in range(H):
                idx = i * H + j
                plt.subplot(H, H, idx + 1)
                plt.bar(classes, p[idx, :].cpu().numpy())
                gt_label = str(self.batch[1].cpu().numpy()[idx])
                plt.title('GT = ' + gt_label)
                plt.xticks(classes, fontsize=8)
                plt.yticks(fontsize=8)
                plt.ylim((0., 1.))
        plt.tight_layout()

        plt.savefig(self.save_dir + '/vis_' + '{:05d}'.format(step) + '.png')
        plt.close()

    def _save_conf_mat_vis(self, step):
        cm = self.conf_mats[-1]
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, vmin=0., vmax=1.)
        plt.xlabel('Predictions')
        plt.ylabel('Targets')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.yticks(rotation='horizontal', fontsize=8)

        plt.savefig(self.save_dir + '/conf_mat_' + '{:05d}'.format(step) + '.png')
