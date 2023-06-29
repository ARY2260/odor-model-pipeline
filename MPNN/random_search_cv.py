import deepchem as dc
from custom_mpnn import CustomMPNNModel
from featurizer import GraphConvConstants
from dataset_mpnn import get_class_imbalance_ratio, get_dataset

def main():
    # Dataset
    all_dataset, _ = get_dataset()

    # Metric
    if args['metric'] == 'roc_auc':
      metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    elif args['metric'] == 'rmse':
      metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    else:
      raise ValueError('Unexpected metric: {}'.format(args['metric']))
    
    # for count_iter in range(num_iter):
    #     # Assembling train and valid datasets
    #     train_folds = all_dataset[:K - count_iter - 1] + all_dataset[K -
    #                                                                  count_iter:]
    #     train_dataset = deepchem.splits.merge_fold_datasets(train_folds)
    #     valid_dataset = all_dataset[K - count_iter - 1]
    
    class_imbalance_ratio = get_class_imbalance_ratio(train_df)

    max_iter = 500
    optimizer = dc.hyper.RandomHyperparamOpt(
                lambda **params: CustomMPNNModel(n_tasks = 6,
                        batch_size=10,
                        learning_rate=0.001,
                        class_imbalance_ratio = class_imbalance_ratio,
                        mode = 'classification',
                        number_atom_features = GraphConvConstants.ATOM_FDIM,
                        number_bond_features = GraphConvConstants.BOND_FDIM,
                        n_classes = 1,
                    **params),
                max_iter=max_iter)
    
    

    params_dict = {"batch_size": [10, 20]}
    transformers = []
    metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                               task_averager=np.mean)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        train_dataset,
        valid_dataset,
        metric,
        transformers,
        use_max=False)
    valid_score = best_model.evaluate(valid_dataset, [metric])