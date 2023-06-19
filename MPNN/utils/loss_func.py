from deepchem.models.losses import Loss


class CustomMultiLabelLoss(Loss):
    """
    Custom Multi-Label Loss function for multi-label classification.

    The cross entropy between two probability distributions.

    The labels should have shape (batch_size) or (batch_size, tasks), and be
    integer class labels.  The outputs have shape (batch_size, classes) or
    (batch_size, tasks, classes) and be logits that are converted to probabilities
    using a softmax function.
    """
    def __init__(self, class_imbalance_ratio=None):
        super(CustomMultiLabelLoss, self).__init__()
        if class_imbalance_ratio is None:
            raise Warning("No class imbalance ratio provided!")
        else:
            import torch
            if not isinstance(class_imbalance_ratio, torch.Tensor):
                raise Exception('class imbalance ratio should be a torch.Tensor')
        self.class_imbalance_ratio = class_imbalance_ratio

    def _create_pytorch_loss(self):
        import torch
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        def loss(output, labels):
            # Convert (batch_size, tasks, classes) to (batch_size, classes, tasks)
            # CrossEntropyLoss only supports (batch_size, classes, tasks)
            # This is for API consistency
            if len(output.shape) == 3:
                output = output.permute(0, 2, 1)

            if len(labels.shape) == len(output.shape):
                labels = labels.squeeze(-1)

            # handle multilabel
            # output shape => (batch_size, classes=1, tasks)
            # binary_output shape => (batch_size, classes=2, tasks) where now we have (1 - probabilities) for ce loss calculation
            probabilities = output[:, 0, :]
            complement_probabilities = 1 - probabilities
            binary_output = torch.stack([probabilities, complement_probabilities], axis=1)

            ce_loss = ce_loss_fn(binary_output, labels.long())
            
            if self.class_imbalance_ratio is None:
                total_loss = ce_loss.sum()
            else:
                if len(self.class_imbalance_ratio) != ce_loss.shape[1]:
                    raise Exception("size of class_imbalance_ratio should be equal to n_tasks")
                balancing_factors = torch.log(1 + self.class_imbalance_ratio)
                balanced_losses = torch.mul(ce_loss, balancing_factors) # loss being weighted by a factor of log(1+ class_imbalance_ratio)
                total_loss = balanced_losses.sum(axis=1) # sum balanced loss across all tasks; shape => (batch_size)
            
            # duplicate loss across all tasks in a batch; shape => (batch_size, n_tasks)
            # This is for API consistency
            return total_loss.unsqueeze(-1).repeat(1,output.shape[-1])

        return loss
