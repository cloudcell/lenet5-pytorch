import matplotlib.pyplot as plt


class TrainLogger(object):
    def __init__(self):
        self.train_loss = []
        self.train_idx = []

        self.val_loss = []
        self.val_acc = []
        self.val_idx = []

    def add_train(self, train_loss, train_idx):
        self.train_idx.append(train_idx)
        self.train_loss.append(train_loss)

    def add_val(self, val_loss, acc, val_idx):
        self.val_idx.append(val_idx)
        self.val_loss.append(val_loss)
        self.val_acc.append(acc)


def plot_history(logger, save_path=None):
    plt.plot(logger.train_idx, logger.train_loss, label='train loss')
    plt.plot(logger.val_idx, logger.val_loss, label='val loss')
    plt.plot(logger.val_idx, logger.val_acc, label='val accuracy')
    plt.xlabel('step')
    plt.legend()
    plt.savefig(save_path, dpi=100)
    plt.close()
