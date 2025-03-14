import torch
from Boosting.boostingUnit import boostingUnit

class boostingClassLayer(torch.nn.Module):

    def __init__(self, in_channels, n_neuron_per_classifier, n_classifier, context_dim, n_prototypes, layer_class, conv,conv_act=lambda x: x,
                 device=None):
        super(boostingClassLayer, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neuron_per_classifier = n_neuron_per_classifier
        self.n_classifier=n_classifier
        self.n_prototypes=n_prototypes
        self.contex_dim = context_dim
        self.conv_act = conv_act
        self.layer_class= layer_class
        self.layer = torch.nn.ModuleList()

        for classifier_index in range(n_classifier):

            self.layer.append(boostingUnit(classifier_index=classifier_index,
                                           in_channels=in_channels,
                                           n_neurons=n_neuron_per_classifier,
                                           context_dim=context_dim,
                                           n_prototypes=n_prototypes,
                                           classifier_class=layer_class,
                                           conv=conv,
                                           conv_act=conv_act,
                                           device=device))

        self.reset_parameters()

    def reset_parameters(self):
        for neuron in self.layer:
            neuron.reset_parameters()

    def set_prototypes(self, training_set,protopytes_indices, dim_test=True):
        for classifier in self.layer:
            classifier.set_prototypes(training_set,protopytes_indices, dim_test)


    def set_opt(self,lr,weight_decay,criterion):
        self.criterion = criterion
        for classifier in self.layer:
            classifier.set_opt(lr,weight_decay,criterion)

    def forward(self, X, context,edge_index):
        H=[]
        for classifier in self.layer:
            H.append(classifier(X,context,edge_index))
        H=torch.cat(H,dim=1)
        return H

    def opt_step(self, h, train_mask,test_mask, valid_mask, context, y, edge_index,epochs, log_path, balanced=False,
                 test_epoch=1, max_n_epochs_without_improvements=30, early_stopping_threshold=0):

        target = torch.where(y == self.layer_class, 1, 0).reshape(-1, 1).expand(-1, 1)

        train_mask = train_mask.to(self.device)
        idx_balanced=train_mask
        if balanced:
            target_pos_idx = torch.masked_select(train_mask, target[train_mask].squeeze().bool())
            target_neg_idx = torch.masked_select(train_mask, (~target[train_mask].squeeze().bool()))

            index_neg_balance = torch.multinomial(target_neg_idx.float(), len(target_pos_idx)).to(self.device)
            idx_balanced = torch.cat((target_pos_idx, index_neg_balance), 0)

        observation_weights = torch.empty(len(idx_balanced)).fill_(1 / len(idx_balanced)).to(self.device)

        layer_loss=0.0
        h_layer=[]
        new_prototypes_indexes=[]
        for classifier in self.layer:
            if len(new_prototypes_indexes) >0:
                classifier.set_prototypes(context[idx_balanced], new_prototypes_indexes)
            loss, h_neuron, observation_weights, alpha, loss_sample=classifier.opt_step(x=h,
                                                                                        train_mask=idx_balanced,
                                                                                        test_mask=test_mask,
                                                                                        valid_mask=valid_mask,
                                                                                        context=context,
                                                                                        y=target,
                                                                                        edge_index=edge_index,
                                                                                        epochs=epochs,
                                                                                        observation_weights=observation_weights,
                                                                                        log_path=log_path,
                                                                                        test_epoch=test_epoch,
                                                                                        max_n_epochs_without_improvements=max_n_epochs_without_improvements,
                                                                                        early_stopping_threshold=early_stopping_threshold)

            layer_loss+=loss
            h_layer.append(h_neuron)
            _,new_prototypes_indexes=torch.topk(loss_sample, self.n_prototypes)


        return layer_loss/self.n_classifier, torch.cat(h_layer,dim=1)