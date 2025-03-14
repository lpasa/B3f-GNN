import os
import os.path as osp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

torch.set_printoptions(threshold=5000)

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from data_reader.LoadCitation import get_idx
from conv.LinearGatedProtoGraphConv import LinearGatedProtoGraphConv

from Boosting.B3FGNN import B3FGNN as MPGLN
from utils.utils_method import printParOnFile

if __name__ == '__main__':
    n_epochs = 200
    n_classes = 7

    n_hidden_layers_list = [1, 2, 3, 4, 5, 6, 7, 8]
    n_neurons_per_classifier_list = [2, 4, 6, 8, 16, 24, 32, 64]
    n_classifiers_per_layer_list =[1, 2, 8, 16, 24]
    n_prototypes_list = [1, 2, 4, 6, 8, 16]

    lr_list = [0.1, 0.2, 0.01, 0.001]
    weight_decay_list = [0, 5e-4, 5e-3]
    balanced_training_list=[True,False]

    test_epoch = 1
    n_run = 2
    max_n_epochs_without_improvements = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DATASET
    dataset_name = 'Cora'
    path = osp.join('~/Dataset/', dataset_name)
    dataset = Planetoid(path, dataset_name, split='full', transform=T.TargetIndegree())
    data = dataset[0].to(device)
    test_type = "B3F-GraphConv-X"

    for run in range(n_run):
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for n_hidden_layers in n_hidden_layers_list:
                    for n_neurons_per_classifier in n_neurons_per_classifier_list:
                        for n_classifiers_per_layer in n_classifiers_per_layer_list:
                            for n_prototypes in n_prototypes_list:
                                for balanced_training in balanced_training_list:

                                    test_name = "run_" + str(run) + "_" + test_type

                                    test_name = test_name + "_data-" + dataset_name + \
                                                "_lr-" + str(lr) \
                                                + "_weight-decay-" + str(weight_decay) + \
                                                "_n_hidden_layers-" + str(n_hidden_layers) \
                                                + "_n_neurons_per_classifier-" + str(n_neurons_per_classifier) \
                                                + "_n_classifiers_per_layer-" + str(n_classifiers_per_layer) \
                                                + "_n_prototypes-" + str(n_prototypes) \
                                                + "_balanced_training-" + str(balanced_training)
                                    print(test_name)
                                    training_log_dir = os.path.join("./test_log/" + test_type, test_name)
                                    if not os.path.exists(training_log_dir):
                                        os.makedirs(training_log_dir)

                                        printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                       par_list={"dataset_name": dataset_name,
                                                                 "learning_rate": lr,
                                                                 "weight_decay": weight_decay,
                                                                 "n_hidden_layers": n_hidden_layers,
                                                                 "n_neurons_per_classifier": n_neurons_per_classifier,
                                                                 "n_classifiers_per_layer": n_classifiers_per_layer,
                                                                 "n_prototypes": n_prototypes,
                                                                 "test_epoch": test_epoch,
                                                                 "balanced_training_list": balanced_training_list})

                                        train_mask, test_mask, val_mask = get_idx(data.x.shape[0], 0.2, run + 1)
                                        model = MPGLN(in_channels=data.x.shape[1],
                                                      n_class=max(data.y) + 1,
                                                      n_neurons_per_classifier=[
                                                                                   n_neurons_per_classifier] * n_hidden_layers,
                                                      n_classifiers_per_layer=[
                                                                                  n_classifiers_per_layer] * n_hidden_layers,
                                                      n_hidden_layers=n_hidden_layers,
                                                      context_dim=data.x.shape[1],
                                                      n_prototypes=n_prototypes,
                                                      conv=LinearGatedProtoGraphConv,
                                                      H_as_context=False,
                                                      device=device)

                                        model.set_networks_layer_protopytes_based_on_x(data, train_mask)

                                        model.set_networks_optimizer(lr=lr, weight_decay=weight_decay,
                                                                     criterion=torch.nn.BCELoss(reduction='none'))

                                        model.train_and_test(data=data,
                                                             train_mask=train_mask,
                                                             test_mask=test_mask,
                                                             valid_mask=val_mask,
                                                             context=data.x.to(device),
                                                             epochs=n_epochs,
                                                             balanced_training=balanced_training,
                                                             test_name=test_name,
                                                             log_path=training_log_dir,
                                                             max_n_epochs_without_improvements=max_n_epochs_without_improvements,
                                                             )
                                        if str(device) == 'cuda':
                                            del model
                                            torch.cuda.empty_cache()

                                    else:
                                        print("test has been already execute")


