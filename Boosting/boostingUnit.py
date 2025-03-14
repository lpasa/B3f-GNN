import torch
import os
import datetime
import time

torch.set_printoptions(profile="full")


def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t loss \t acc \t avg_epoch_time \n")

    return train_log, test_log, valid_log


class boostingUnit(torch.nn.Module):

    def __init__(self, classifier_index, in_channels, n_neurons, context_dim, n_prototypes, classifier_class, conv, conv_act=lambda x: x, device=None):
        super(boostingUnit, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.classifier_index=classifier_index
        self.in_channels = in_channels
        self.n_neurons=n_neurons
        self.n_prototypes = n_prototypes
        self.contex_dim = context_dim
        self.conv_act = conv_act
        self.classifier_class = classifier_class

        self.alpha = torch.tensor(1)
        self.alpha.requires_grad=False

        if 'Hyper' in conv.__name__:
            self.neuron = conv(in_channels=in_channels,
                               out_channels=n_neurons,
                               context_dim=context_dim,
                               n_hyperplanes=n_prototypes,
                               device=device)
        else:
            self.neuron=conv(in_channels=in_channels,
                             out_channels=n_neurons,
                             context_dim=context_dim,
                             n_prototypes=n_prototypes,
                             device=device)
        self.output_mixer=torch.nn.Linear(in_features=n_neurons,
                                          out_features=1).to(device)
        self.out_fun = torch.nn.Sigmoid()
        self.reset_parameters()


    def reset_parameters(self):
        self.neuron.reset_parameters()

    def set_prototypes(self, training_set, protopytes_indices=None, dim_test=True):
        self.neuron.set_prototypes(training_set, protopytes_indices, dim_test)



    def forward(self, X, context, edge_index):

        h=self.neuron(X,context,edge_index)
        h=self.output_mixer(h)
        out = self.alpha*h

        return out

    def forward_neurons(self, X, context, edge_index):

        h=self.neuron(X,context,edge_index)
        out = self.alpha*h

        return out


    def set_opt(self,lr,weight_decay,criterion):
        self.optimizer_neurons = torch.optim.RMSprop(self.neuron.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_outputmixer = torch.optim.RMSprop(self.output_mixer.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion=criterion


    def opt_step(self, x, train_mask, test_mask, valid_mask, context, y,edge_index, epochs,observation_weights,log_path=".",test_epoch=1,
                 max_n_epochs_without_improvements=30, early_stopping_threshold=0):

        train_log, test_log, valid_log = prepare_log_files("classifier_{:d}_class_{:d}".format(self.classifier_index, self.classifier_class), log_path)
        print("classifier {:d} - class {:d} -- optimization process".format(self.classifier_index, self.classifier_class))

        epoch_time_sum = 0
        n_epochs_without_improvements = 0
        best_loss_so_far = -1.0

        for e in range(epochs):
            epoch_start_time = time.time()
            self.train()

            self.optimizer_neurons.zero_grad()

            h_neurons=self.forward_neurons(x, context, edge_index)
            h_neuron_mask = h_neurons[train_mask]

            pred_neuron_c = self.out_fun(h_neuron_mask)

            target_neurons = y[train_mask]
            target_neurons= target_neurons.expand(-1, self.n_neurons)
            target_neurons = target_neurons.float()

            loss_neurons_samples = self.criterion(pred_neuron_c, target_neurons)
            boosting_neurons_loss = torch.einsum('i,ij->ij', observation_weights, loss_neurons_samples)
            boosting_neurons_loss = torch.mean(boosting_neurons_loss)

            loss = torch.mean(boosting_neurons_loss)

            loss.backward()
            self.optimizer_neurons.step()

            self.optimizer_outputmixer.zero_grad()
            h = self.forward(x, context, edge_index)
            h_mask = h[train_mask]


            pred_c = self.out_fun(h_mask)

            target = y[train_mask]
            target=target.float()

            loss_samples = self.criterion(pred_c, target)
            boosting_loss = torch.einsum('i,ij->ij', observation_weights, loss_samples)
            boosting_loss = torch.mean(boosting_loss)

            loss = torch.mean(loss_samples)

            loss.backward()
            self.optimizer_outputmixer.step()

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if e % test_epoch == 0:
                print("epoch : ", e, " -- loss: ", loss.item())

                acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_layer(x, edge_index,
                                                                                                         y, train_mask,
                                                                                                         context)
                acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_layer(x, edge_index, y,
                                                                                                     test_mask,
                                                                                                     context)
                acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_layer(x, edge_index,
                                                                                                         y, valid_mask,
                                                                                                         context)
                print(" -- training acc : ", (acc_train_set, correct_train_set, n_samples_train_set),
                      " -- test_acc : ", (acc_test_set, correct_test_set, n_samples_test_set),
                      " -- valid_acc : ", (acc_valid_set, correct_valid_set, n_samples_valid_set))
                print("------")

                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        e,
                        loss_train_set,
                        acc_train_set,
                        epoch_time_sum / test_epoch,
                    ))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        e,
                        loss_test_set,
                        acc_test_set,
                        epoch_time_sum / test_epoch,
                    ))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        e,
                        loss_valid_set,
                        acc_valid_set,
                        epoch_time_sum / test_epoch,
                    ))

                valid_log.flush()

                if loss_valid_set < best_loss_so_far or best_loss_so_far == -1:
                    best_loss_so_far = loss_valid_set
                    n_epochs_without_improvements = 0
                    best_epoch = e
                    print("--ES--")
                    print("new_best_model, with loss:", best_loss_so_far.item())
                    print("------")

                elif loss_valid_set >= best_loss_so_far + early_stopping_threshold:
                    n_epochs_without_improvements += 1
                else:
                    n_epochs_without_improvements = 0

                if n_epochs_without_improvements >= max_n_epochs_without_improvements or e == epochs - 1:
                    print("___Early Stopping at epoch ", best_epoch, "____")
                    break
                epoch_time_sum = 0

        h = self.forward(x, context, edge_index)
        h_mask = h[train_mask]

        pred_c = self.out_fun(h_mask)

        bin_err=torch.abs(torch.round(pred_c)-target)
        err = (torch.sum(torch.einsum('i,ij->ij', observation_weights, bin_err))/torch.sum(observation_weights)).detach()
        if err>0 and err<1:
            self.alpha = torch.log((1-err)/err)
            observation_weights=torch.exp(self.alpha*bin_err).squeeze() * observation_weights

        return boosting_loss, h.detach(), observation_weights.detach(), self.alpha.detach(), loss_samples.T[0]

    def eval_layer(self, h, edge_index, y, mask, context):
        self.eval()
        h_out = self.forward(h, context, edge_index)
        pred = self.out_fun(h_out[mask])
        target = y[mask]
        n_samples = len(target)
        round_pred = torch.round(pred)
        correct = round_pred.eq(target).sum().item()
        acc = correct / n_samples
        loss = torch.mean(self.criterion(pred, target.float()))

        return acc, correct, n_samples, loss / n_samples