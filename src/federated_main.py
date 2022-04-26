import os
import sys
import copy
import time
import pickle
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, get_logger

from trainLocalNetwork import TrainLocalNetwork, get_global_model

path_project = os.path.abspath('..')

if __name__ == '__main__':

    args = args_parser()
    model_name = '{}_{}'.format(args.model_name, args.JOBID)
    log_dir = '../training_result/{}_time_{}'.format(model_name, time.strftime('%Y%m%d_%H%M%S'))
    logger = get_logger(log_dir)

    save_tbx_log = log_dir + '/tensorboard_log'

    writer = SummaryWriter(save_tbx_log)

    model_path = os.path.join(args.model_path, model_name)
    if model_path is not None and not os.path.exists(model_path):
        os.makedirs(model_path)

    best_model_path = os.path.join(model_path, 'model_best.pth.tar')
    checkpoint_path = os.path.join(model_path, 'checkpint.pth.tar')

    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)
    device = torch.device('cuda:0')

    cudnn.enabled = True
    cudnn.benchmark = True

    sub_list = os.listdir(args.data_dir)
    args.num_users = len(sub_list)
    user_groups = {}
    for (ind,subgrid_dir) in enumerate(sub_list):
        user_groups[ind] = TrainLocalNetwork(args, subgrid_dir)

    # BUILD MODEL
    global_model = get_global_model(user_groups[0])

    # Set the model to train and send it to device.
    global_model.to(device)
    # global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    best_loss = 1000000

    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        logger.info(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        time_1 = time.time()
        for idx in idxs_users:
            # local_model = copy.deepcopy(user_groups[idx])
            local_model = user_groups[idx]
            w, loss = local_model.run_train(copy.deepcopy(global_model), epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        time_2 = time.time()
        logger.info('epoch {}: {} clients cost {} seconds'.format(epoch, m, time_2-time_1))
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        logger.info("\t - train_avg_loss : {:.4}".format(loss_avg))

        writer.add_scalars('Train', {'loss': loss_avg}, epoch)

        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        val_loss = []
        F1_list = []
        for c in range(args.num_users):
            local_model = user_groups[c]
            _val_loss, prec, recall, F1, = local_model.run_test(global_model)
            logger.info("\t - client {}: val_loss={:.4}, prec={:.4}, recall={:.4}, F1={:.4}".format(c, _val_loss, prec, recall, F1))
            val_loss.append(_val_loss)
            F1_list.append(F1)
            writer.add_scalars('Val - Client {}'.format(c),
                               {'val_loss': _val_loss,
                                'Precision': prec,
                                'Recall': recall,
                                'F1': F1}, epoch)


        F1_avg = sum(F1_list)/len(F1_list)
        val_loss_avg = sum(val_loss)/len(val_loss)
        writer.add_scalars('Val_loss_avg',
                           {'val_avg_loss': val_loss_avg}, epoch)

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg

            torch.save({'epoch': epoch + 1,
                        'model_state': global_model.state_dict()}, best_model_path)

        torch.save({'epoch': epoch + 1,
                    'model_state': global_model.state_dict()}, checkpoint_path)

    writer.close()



