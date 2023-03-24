def MSE(pred_mu, label):
    return ((pred_mu - label) ** 2).mean()


def entropy(pred_dist):
    return (pred_dist.entropy()).mean()


def log_prob(pred_dist, label):
    return (pred_dist.log_prob(label)).mean()
