# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from driver.DataLoader import create_batch_iter, pair_data_variable


def train(model, train_data, dev_data, test_data, vocab_srcs, vocab_tgts, config):
    # optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config.learning_algorithm == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer method: ' + config.learning_algorithm)

    # train
    global_step = 0
    best_acc = 0
    print('\nstart training...')
    for iter in range(config.epochs):
        iter_start_time = time.time()
        print('Iteration: ' + str(iter))

        batch_num = int(np.ceil(len(train_data) / float(config.batch_size)))
        batch_iter = 0
        for batch in create_batch_iter(train_data, config.batch_size, shuffle=True):
            start_time = time.time()
            feature, target, lengths, mask = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)
            model.train()
            optimizer.zero_grad()
            logit = model(feature, lengths, mask)

            loss = F.cross_entropy(logit, target)
            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()

            correct = (torch.max(logit, 1)[1].view(target.size()) == target).sum().item()
            accuracy = 100.0 * correct / target.size()[0]

            during_time = float(time.time() - start_time)
            print("Step:{}, Iter:{}, batch:{}, accuracy:{:.4f}({}/{}), time:{:.2f}, loss:{:.6f}"
                  .format(global_step, iter, batch_iter, accuracy, correct, target.size()[0], during_time, loss_value))

            batch_iter += 1
            global_step += 1

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                if dev_data is not None:
                    dev_acc = evaluate(model, dev_data, global_step, vocab_srcs, vocab_tgts, config)
                test_acc = evaluate(model, test_data, global_step, vocab_srcs, vocab_tgts, config)
                if dev_data is not None:
                    if dev_acc > best_acc:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_acc))
                        best_acc = dev_acc
                    if -1 < config.save_after <= iter:
                        torch.save(model.state_dict(), os.path.join(config.model_path, 'model.' + str(global_step)))
                else:
                    if test_acc > best_acc:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, test_acc))
                        best_acc = test_acc
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.model_path, 'model.' + str(global_step)))
        during_time = float(time.time() - iter_start_time)
        print('one iter using time: time:{:.2f}'.format(during_time))


def evaluate(model, data, step, vocab_srcs, vocab_tgts, config):
    model.eval()
    start_time = time.time()
    corrects, size = 0, 0

    f_labels = [vocab_tgts.i2w[i] for i in range(len(vocab_tgts.i2w))]
    # 每个标签预测正确的
    f_corrects = [0 for _ in range(len(vocab_tgts.i2w))]
    # 每个标签预测的个数
    f_predicts = [0 for _ in range(len(vocab_tgts.i2w))]
    # 实际上每个标签的数量
    f_size = [0 for _ in range(len(vocab_tgts.i2w))]

    for batch in create_batch_iter(data, config.batch_size):
        feature, target, lengths, mask = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)
        logit = model(feature, lengths, mask)
        correct = (torch.max(logit, 1)[1].view(target.size()) == target).sum().item()
        corrects += correct
        size += target.size()[0]

        for i in range(len(f_labels)):
            f_size[i] += (target == i).sum().item()
            f_predicts[i] += (torch.max(logit, 1)[1].view(target.size()) == i).sum().item()
            f_corrects[i] += (torch.mul(
                torch.max(logit, 1)[1].view(target.size()) == i, target == i)).sum().item()
    accuracy = 100.0 * corrects / size
    during_time = float(time.time() - start_time)
    print("\nevaluate result: ")
    print("Step:{}, accuracy:{:.4f}({}/{}), time:{:.2f}".format(step, accuracy, corrects, size, during_time))
    is_ok = True
    for i in range(len(f_labels)):
        if f_predicts[i] == 0:
            is_ok = False
            break
        if f_corrects[i] == 0:
            is_ok = False
            break
    if is_ok:
        recall = [0 for _ in range(len(f_labels))]
        precision = [0 for _ in range(len(f_labels))]
        f1 = [0 for _ in range(len(f_labels))]
        for i in range(len(f_labels)):
            recall[i] = 100.0 * float(f_corrects[i] / f_size[i])
            precision[i] = 100.0 * float(f_corrects[i] / f_predicts[i])
            f1[i] = 2.0 / ((1.0 / recall[i]) + (1.0 / precision[i]))

            # print('\npolarity: {}  corrects: {}  predicts: {}  size: {}'.format(f_labels[i], f_corrects[i],
            #                                                                     f_predicts[i], f_size[i]))
            # print('polarity: {}  recall: {:.4f}%  precision: {:.4f}%  f1: {:.4f}% \n'.format(f_labels[i], recall[i],
            #                                                                                  precision[i], f1[i]))

        aver_p = 0
        aver_r = 0
        aver_f1 = 0
        for i in range(len(f_labels)):
            aver_p += precision[i]
            aver_r += recall[i]
            aver_f1 += f1[i]
        print("precision: ", aver_p / len(f_labels))
        print("recall: ", aver_r / len(f_labels))
        print("macro f1: ", aver_f1 / len(f_labels))

    model.train()
    return accuracy


def predict(model, data, vocab_srcs, vocab_tgts, config):
    model.eval()
    start_time = time.time()
    corrects, size = 0, 0

    f_labels = [vocab_tgts.i2w[i] for i in range(len(vocab_tgts.i2w))]
    # 每个标签预测正确的
    f_corrects = [0 for _ in range(len(vocab_tgts.i2w))]
    # 每个标签预测的个数
    f_predicts = [0 for _ in range(len(vocab_tgts.i2w))]
    # 实际上每个标签的数量
    f_size = [0 for _ in range(len(vocab_tgts.i2w))]

    for batch in create_batch_iter(data, config.batch_size):
        feature, target, starts, ends, feature_lengths = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)
        logit = model(feature, feature_lengths, starts, ends)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum().item()
        corrects += correct
        size += len(batch)

        for i in range(len(f_labels)):
            f_size[i] += (target == i).sum().item()
            f_predicts[i] += (torch.max(logit, 1)[1].view(target.size()) == i).sum().item()
            f_corrects[i] += (torch.mul(
                torch.max(logit, 1)[1].view(target.size()).data == i, target == i)).sum().item()
    accuracy = 100.0 * corrects / size
    during_time = float(time.time() - start_time)
    print("\nevaluate result: ")
    print("accuracy:{:.4f}({}/{}), time:{:.2f}".format(accuracy, corrects, size, during_time))
    is_ok = True
    for i in range(len(f_labels)):
        if f_predicts[i] == 0:
            is_ok = False
            break
        if f_corrects[i] == 0:
            is_ok = False
            break
    if is_ok:
        recall = [0 for _ in range(len(f_labels))]
        precision = [0 for _ in range(len(f_labels))]
        f1 = [0 for _ in range(len(f_labels))]
        for i in range(len(f_labels)):
            recall[i] = 100.0 * float(f_corrects[i] / f_size[i])
            precision[i] = 100.0 * float(f_corrects[i] / f_predicts[i])
            f1[i] = 2.0 / ((1.0 / recall[i]) + (1.0 / precision[i]))

            print('\npolarity: {}  corrects: {}  predicts: {}  size: {}'.format(f_labels[i], f_corrects[i],
                                                                                f_predicts[i], f_size[i]))
            print('polarity: {}  recall: {:.4f}%  precision: {:.4f}%  f1: {:.4f}% \n'.format(f_labels[i], recall[i],
                                                                                             precision[i], f1[i]))

        aver_p = 0
        aver_r = 0
        aver_f1 = 0
        for i in range(len(f_labels)):
            aver_p += precision[i]
            aver_r += recall[i]
            aver_f1 += f1[i]
        print("precision: ", aver_p / 3.0)
        print("recall: ", aver_r / 3.0)
        print("macro f1: ", aver_f1 / 3.0)