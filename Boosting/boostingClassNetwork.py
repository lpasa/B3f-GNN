import os
import torch
from Boosting.bostingLayer import boostingClassLayer


class classNetwork(torch.nn.Module):
    def __init__(self, in_channels, n_neurons_per_classifier, n_classifiers_per_layer, n_hidden_layers, context_dim,
                 n_prototypes, network_class, conv, conv_act, device=None, H_as_context = False):
        super(classNetwork, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neurons_per_classifier=n_neurons_per_classifier
        self.n_classifiers_per_layer=n_classifiers_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.n_prototypes = n_prototypes
        self.contex_dim = context_dim
        self.network_class=network_class
        self.layers = torch.nn.ModuleList()
        self.conv_act=conv_act
        self.H_as_context=H_as_context

        assert n_hidden_layers == len(n_classifiers_per_layer) and n_hidden_layers == len(n_neurons_per_classifier), "Error on input parameters"

        self.layers.append(
            boostingClassLayer(in_channels=in_channels,
                               n_neuron_per_classifier=n_neurons_per_classifier[0],
                               n_classifier=n_classifiers_per_layer[0],
                               context_dim=context_dim,
                               n_prototypes=n_prototypes,
                               layer_class=network_class,
                               conv=conv,
                               conv_act=conv_act,
                               device=device))
        self.protopytes_indices = [None]

        for layer_index in range(1, n_hidden_layers):

            if H_as_context is True:
                self.layers.append(boostingClassLayer(in_channels=n_classifiers_per_layer[layer_index-1],
                                                      n_neuron_per_classifier=n_neurons_per_classifier[layer_index],
                                                      n_classifier=n_classifiers_per_layer[layer_index],
                                                      context_dim=n_classifiers_per_layer[layer_index-1],
                                                      n_prototypes=n_prototypes,
                                                      layer_class=network_class,
                                                      conv=conv,
                                                      conv_act=conv_act,
                                                      device=device))
            else:
                self.layers.append(boostingClassLayer(in_channels=n_classifiers_per_layer[layer_index-1],
                                                      n_neuron_per_classifier=n_neurons_per_classifier[layer_index],
                                                      n_classifier=n_classifiers_per_layer[layer_index],
                                                      context_dim=context_dim,
                                                      n_prototypes=n_prototypes,
                                                      layer_class=network_class,
                                                      conv=conv,
                                                      conv_act=conv_act,
                                                      device=device))
            self.protopytes_indices.append(None)


        if H_as_context is True:
            self.layers.append(boostingClassLayer(in_channels=n_classifiers_per_layer[-1],
                                                  n_neuron_per_classifier=n_neurons_per_classifier[-1],
                                                  n_classifier=1,
                                                  context_dim=n_classifiers_per_layer[-1],
                                                  n_prototypes=n_prototypes,
                                                  layer_class=network_class,
                                                  conv=conv,
                                                  conv_act=conv_act,
                                                  device=device))
        else:
            self.layers.append(boostingClassLayer(in_channels=n_classifiers_per_layer[-1],
                                                  n_neuron_per_classifier=n_neurons_per_classifier[-1],
                                                  n_classifier=1,
                                                  context_dim=context_dim,
                                                  n_prototypes=n_prototypes,
                                                  layer_class=network_class,
                                                  conv=conv,
                                                  conv_act=conv_act,
                                                  device=device))

        self.protopytes_indices.append(None)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def set_layers_prototypes_base_on_input_rep(self, data, mask,fix_protopytes_indices=False):
        for l,layer in enumerate(self.layers):
            if fix_protopytes_indices:
                if self.protopytes_indices[l] is None:
                    self.protopytes_indices[l]= torch.randint(low=0, high=data.x[mask].shape[0], size=(self.n_neuron[l],self.n_prototypes))
            layer.set_prototypes(data.x, self.protopytes_indices[l], dim_test=False)

    def set_layers_prototypes_base_on_hidden_rep(self, data, mask, context, fix_protopytes_indices=False):
        X = data.x
        edge_index = data.edge_index
        h = X
        h=h.to(self.device)
        for l,layer in enumerate(self.layers):
            if fix_protopytes_indices:
                if self.protopytes_indices[l] is None:
                    self.protopytes_indices[l]= torch.randint(low=0, high=X[mask].shape[0], size=(self.n_neuron[l],self.n_prototypes))

            layer.set_prototypes(h[mask],self.protopytes_indices[l])
            if context is None or self.H_as_context:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)

    def set_layers_prototypes_locally_base_on_hidden_rep(self, data, mask, context, fix_protopytes_indices=False):
        X = data.x
        edge_index = data.edge_index
        h = X
        h = h.to(self.device)
        for l, layer in enumerate(self.layers):
            h_train=h[mask]
            seed_proto = torch.randint(low=0, high=h_train.shape[0],size=[self.n_neuron[l]])
            seed_dists = torch.cdist(h_train[seed_proto],h_train,p=2)
            prototypes_prob_dist = torch.distributions.Categorical(seed_dists)
            protopytes_indices = prototypes_prob_dist.sample([self.n_prototypes]).T

            layer.set_prototypes(h[mask], protopytes_indices)
            if context is None or self.H_as_context:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)

    def set_layers_optimizer(self,lr,weight_decay,criterion):
        for layer in self.layers:
            layer.set_opt(lr=lr, weight_decay=weight_decay,criterion=criterion)

    def _set_layer_prototypes_locally_base_on_hidden_rep(self, layer, l, h, mask):


        h_train=h[mask]
        seed_proto = torch.randint(low=0, high=h_train.shape[0],size=[self.n_neuron[l]])
        seed_dists = torch.cdist(h_train[seed_proto],h_train,p=2)
        if torch.sum(seed_dists)>0:
            prototypes_prob_dist = torch.distributions.Categorical(seed_dists)
            protopytes_indices = prototypes_prob_dist.sample([self.n_prototypes]).T
        else:
            protopytes_indices= torch.randint(low=0, high=h_train.shape[0], size=(self.n_neuron[l],self.n_prototypes))
        layer.set_prototypes(h[mask], protopytes_indices)


    def forward(self, data,context):
        X = data.x
        edge_index = data.edge_index

        h=X
        h=h.to(self.device)
        layer_classification =[]
        for layer in self.layers:
            if context is None or self.H_as_context:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)
            layer_classification.append(h)

        return layer_classification,h

    def layers_optimization_step(self,data,train_mask, test_mask, valid_mask,context,epochs,
                         log_path=".", max_n_epochs_without_improvements=30, early_stopping_threshold=0,
                                 test_epoch=1, set_layer_proto_dynamically=False, balanced_training=False):

        X = data.x.to(self.device)
        edge_index = data.edge_index
        y=torch.squeeze(data.y)
        h = X
        h=h.to(self.device)

        for l,layer in enumerate(self.layers):

            if set_layer_proto_dynamically:
                self._set_layer_prototypes_locally_base_on_hidden_rep(layer=layer,
                                                                      l=l,
                                                                      h=h,
                                                                      mask=train_mask)

            layer.train()

            print("--- Running: layer {:d}, class {:d} ".format(l,self.network_class))
            if context is None or self.H_as_context:
                cur_context =h.to(self.device)
            else:
                cur_context = context.to(self.device)

            loss,h_tm1=layer.opt_step(h, train_mask, test_mask, valid_mask, cur_context, y, edge_index, epochs,
                                      log_path, balanced_training,test_epoch, max_n_epochs_without_improvements,
                                      early_stopping_threshold)
            h = h_tm1

        return h


    def save_model(self,test_name, log_folder='./'):
        torch.save(self.state_dict(), os.path.join(log_folder,test_name+'.pt'))