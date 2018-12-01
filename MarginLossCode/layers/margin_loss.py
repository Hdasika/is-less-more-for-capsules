import tensorflow as tf


def margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0):
    '''
    Args:
        margin: scalar, the margin after subtracting 0.5 from raw_logits.
        downweight: scalar, the factor for negative cost.
    '''

    def _margin_loss(labels, raw_logits):
        """Penalizes deviations from margin for each logit.

        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.

        Args:
        labels: tensor, one hot encoding of ground truth.
        raw_logits: tensor, model predictions in range [0, 1]


        Returns:
        A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = pos_weight * labels * tf.cast(tf.less(logits, margin),
                                       tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
          tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    return _margin_loss

