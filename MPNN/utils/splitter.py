from skmultilearn.model_selection import IterativeStratification
import tempfile
import pandas as pd
import deepchem as dc


def iterative_train_test_split(dataset, test_size, random_state=None, train_dir=None, test_dir=None):
    """Iteratively stratified train/test split

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    random_state : None | int | np.random.RandomState
        the random state seed (optional)

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """
    X, y = pd.DataFrame(dataset.X), pd.DataFrame(dataset.y)
    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
        # shuffle=True,
        random_state=random_state,
    )
    train_indexes, test_indexes = next(stratifier.split(X, y))
    if train_dir is None:
        train_dir = tempfile.mkdtemp()
    # if valid_dir is None:
    #     valid_dir = tempfile.mkdtemp()
    if test_dir is None:
        test_dir = tempfile.mkdtemp()
    train_dataset = dataset.select(train_indexes.tolist(), train_dir)
    # valid_dataset = dataset.select(valid_inds, valid_dir)
    test_dataset = dataset.select(test_indexes.tolist(), test_dir)
    if isinstance(train_dataset, dc.data.DiskDataset):
        train_dataset.memory_cache_size = 40 * (1 << 20)  # 40 MB
    return train_dataset, test_dataset
