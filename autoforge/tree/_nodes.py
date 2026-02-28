class TreeNode:
    def __init__(
        self,
        data_left=None,
        data_right=None,
        best_feature=None,
        best_threshold=None,
        information_gain=None,
        prob=None,
        var_red=None,
    ):
        self.left = data_left
        self.right = data_right
        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.info_gain = information_gain
        self.pred_prob = prob
        self.var_red = var_red
