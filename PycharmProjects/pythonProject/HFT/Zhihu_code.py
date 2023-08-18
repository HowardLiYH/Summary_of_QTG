from args import args
from plistlib import load
from re import T
import torch
import torch.nn as nn
import stock
# with open('/home/ubuntu/55data_influx.pkl', "rb") as fh:
# raw3 = pickle.load(fh)

import numpy as np
import tqdm
import logger
import wandb
import models
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
epochs = 1000
lr = args.lr
from stock import StockEnv, abs_profit, mean_profit, std_profit
import datetime

args.temp = 1
max_r2 = 0
max_r2_itr = -1
min_l = 0
min_l_itr = -1
max_test2_r2 = 0
max_test2_r2_itr = -1
corr_test2_r2 = 0
max_p = -1
max_p_itr = -1
corr_p = 0
from numba import jit
import random
import numpy as np
import torch

if args.set_seed:
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)


def mean(l):
    return sum(l) / len(l)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


import pandas as pd

interval = int(1e3)


def stabalize_bn(model, dataset, test_field="", argmax=False, r=1):
    model.train()
    loader_seq = torch.utils.data.DataLoader(dataset, shuffle=False,
                                             sampler=torch.utils.data.SequentialSampler(dataset),
                                             batch_size=args.batch_size, num_workers=4, pin_memory=True)
    for i, (x, y, index, mask) in tqdm.tqdm(enumerate(loader_seq), ascii=True, total=len(loader_seq)):
        x = x.cuda()
        model.get_action0(x, argmax=argmax, r=r)
        model.get_action1(x, argmax=argmax, r=r)


def calculate_train_RLd3_35(model, dataset, test_field="", argmax=False, r=1):
    model.eval()
    loader_seq = torch.utils.data.DataLoader(dataset, shuffle=False,
                                             sampler=torch.utils.data.SequentialSampler(dataset),
                                             batch_size=args.batch_size, num_workers=4, pin_memory=True)
    actions = torch.zeros((dataset.data_X.size(0), args.action_size, r))
    actions2 = torch.zeros((dataset.data_X.size(0), r))
    actions2_shrink = torch.zeros((len(dataset), r))
    logprobs = torch.zeros((dataset.data_X.size(0), r))
    logprobs2 = torch.zeros((dataset.data_X.size(0), r))

    with torch.no_grad():
        start, end = 0, 0
        for i, (x, y, index, mask) in tqdm.tqdm(enumerate(loader_seq), ascii=True, total=len(loader_seq)):
            end += x.size(0)
            x = x.cuda()
            action, logprob = model.get_action0(x, argmax=argmax, r=r)
            action2, logprob2 = model.get_action1(x, argmax=argmax, r=r)
            actions[index] = action.cpu().detach()
            logprobs[index] = logprob.cpu().detach()
            actions2[index] = action2.cpu().detach().float()
            actions2_shrink[start:end] = action2.cpu().detach().float()
            logprobs2[index] = logprob2.cpu().detach()
            start = end
    if argmax:
        for i in range(args.d_num):
            wandb.log({test_field + "/closeprop_" + str(args.d_list[i]): (actions2_shrink == i).sum() / len(dataset)},
                      step=epoch)
    print("before?")
    from influxdb import DataFrameClient
    # if args.write_db:
    #     client = DataFrameClient(args.ip, ****, '**', '********', 'signal')
    #     data_1 = pd.DataFrame()
    #     time_index = np.load(str(args.profit_type)+"dout4time"+"_"+";".join(args.test)+"std"+".npy")
    #     tmp = transAct(actions).cpu().detach().squeeze().numpy()
    #     tmp[tmp>10000] = 10000
    #     tmp2 = transAct(actions2, "7").cpu().detach().squeeze().numpy()
    #     data_1['din'] = pd.DataFrame(tmp)
    #     data_1['dout'] = pd.DataFrame(tmp2)
    #     data_1['time'] = pd.DataFrame(time_index)
    #     data_1.index = data_1.time
    #     i = 0
    #     print(len(data_1))
    #     for i in range(1, int(len(data_1.index)//1e5)):
    #         print("after?")
    #         print((i-1)*int(1e5), i*int(1e5))
    #         client.write_points(data_1[['din']][(i-1)*int(1e5):i*int(1e5)], "backtest_din"+args.type, {})
    #         client.write_points(data_1[['dout']][(i-1)*int(1e5):i*int(1e5)], "backtest_dout"+args.type, {})
    #     print((i-1)*int(1e5), len(data_1))
    #     client.write_points(data_1[['din']][i*int(1e5):],  "backtest_din"+args.type, {})
    #     client.write_points(data_1[['dout']][i*int(1e5):],  "backtest_dout"+args.type, {})
    return dataset.data_X, actions, actions2, logprobs, logprobs2


def train(model, loader, opt, epoch, old):
    global max_r2, max_r2_itr, min_l, min_l_itr, max_test2_r2_itr, max_test2_r2, corr_test2_r2, max_p, max_p_itr, corr_p
    model.train()
    pg_loss_meter = logger.AverageMeter("pg_loss", ":.3f")
    ratio_meter = logger.AverageMeter("ratio", ":.3f")
    opt.zero_grad()
    l = [pg_loss_meter, ratio_meter]
    progress = logger.ProgressMeter(len(loader), l, prefix=f"Train Epoch: [{epoch}]")
    num_updates = 0
    for i, (b_obs, b_logprobs, b_actions, b_returns) in tqdm.tqdm(enumerate(loader), ascii=True, total=len(loader)):
        model.train()

        b_obs, b_logprobs, b_actions, b_returns = b_obs.cuda(), b_logprobs.cuda(), b_actions.cuda(), b_returns.cuda()
        if old == 0:
            _, newlogprob = model.get_action0(b_obs, b_actions)
        elif old == 1:
            _, newlogprob = model.get_action1(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = torch.clamp(logratio, max=50).exp()
        pg_loss1 = -b_returns * ratio
        pg_loss2 = -b_returns * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        loss = pg_loss
        pg_loss_meter.update(pg_loss.item(), b_obs.size(0))
        ratio_meter.update(ratio.mean().item(), b_obs.size(0))
        wandb.log({"train" + str(old) + "/pg_loss": pg_loss_meter.avg}, step=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        if (i + 1) % args.gradient_acc_step == 0:
            if args.div_grad_acc:
                for name, param in model.named_parameters():
                    param.grad.data = param.grad.data / args.gradient_acc_step
            opt.step()
            print("step")
            opt.zero_grad()
            num_updates += 1
            if num_updates >= args.updates_thres:
                break
        if "RL" in args.arch:
            if i % 200 == 0:
                progress.display(i)


model = models.__dict__[args.arch]()
model = model.cuda()
print("number of parameters:", count_parameters(model))
print(model)

# model = torch.load("results/best_model_RLTSTransformerEncoderRLSeq3_10_21-10_25_10_21-10_25_10_21-10_251670915456.9008617lag11.pt")

bn_params = [v for n, v in list(model.named_parameters()) if ("bn" in n) and v.requires_grad]
rest_params = [v for n, v in list(model.named_parameters()) if ("bn" not in n) and v.requires_grad]

opt = torch.optim.AdamW([
    {
        "params": bn_params,
        "weight_decay": 0 if args.no_bn_decay else args.wd,
    },
    {"params": rest_params, "weight_decay": args.wd},
], lr=lr, weight_decay=args.wd)

from random import randrange
import time

args.rand_number = str(time.time())

import gym


def transAct(preaction, type_=None):
    if type_ == "7":
        # action = preaction.detach().clone()
        # if args.no_order:
        #     action[action==args.d_num-1] = 1e5
        # action = action + args.offset

        action = preaction.detach().clone()
        # print(len(stock.from_list), args.d_num, args.d_list)
        for i in range(len(args.d_list)):
            action[preaction == i] = args.d_list[i]

        for i in range(len(args.d_list)):
            print((action == args.d_list[i]).sum() / action.size(0))
    else:
        if args.clamp_neg:
            action = torch.clamp(preaction, min=0)
        elif args.sigmoid_neg:
            action = torch.sigmoid(preaction) * args.sig_alpha
        elif args.exp_neg:
            # print("why", preaction)
            action = torch.exp(preaction)
            # print(preaction[119896])
            # print(action[119896])
            # print("why", action)

            # action = torch.cat([torch.ones_like(action)*4, torch.ones_like(action)*4], dim=1)\
            # action = torch.cat([action, torch.ones_like(action)*4], dim=1)
        else:
            action = preaction
    return action


@jit
def calculate_profit_with_chosen(valid_sellpoint, chosen_value_total, position=args.bin, lag=1, lag2=1):
    # print(valid_sellpoint)
    chosen_vt = chosen_value_total
    positions_dict = [0 for i in range(len(valid_sellpoint))]
    # profit_positions = [0 for i in range(len(valid_sellpoint))]
    # addedvb_positions = [[] for i in range(len(valid_sellpoint))]

    valid_buypoint = (chosen_vt != 0).nonzero()[0]
    # for i in range(len(chosen_vt)):
    # print(chosen_vt[i])
    profit_position = np.zeros(len(chosen_vt))

    def get_loc(vb):
        for i in range(len(valid_sellpoint)):
            if vb < valid_sellpoint[i]:
                return i, False
            if valid_sellpoint[i] <= vb and vb < valid_sellpoint[i] + lag + lag2 - 1:
                return i, True

    total_profit = 0
    # print(len(valid_buypoint))
    deferred_positions = [(0, 0)]
    deferred_positions = deferred_positions[1:]
    clear_list = []
    # print(len(valid_buypoint), len(valid_sellpoint))
    for vb in valid_buypoint:
        # print(vb)
        # print(total_profit, np.sum(profit_position))
        # if len(clear_list) != 0:
        # print(sum(clear_list)/len(clear_list), len(clear_list), len(valid_buypoint), len(valid_sellpoint))
        i = 0
        retained_items = []
        for i in range(len(deferred_positions)):
            if vb >= deferred_positions[i][1]:
                positions_dict[deferred_positions[i][0]] += 1
            else:
                retained_items.append(deferred_positions[i])
        deferred_positions = retained_items
        # print(positions_dict[:10], max(positions_dict), deferred_positions, lag)
        if vb < valid_sellpoint[-1]:
            bin, atpoint = get_loc(vb)
            if not atpoint:
                if positions_dict[bin] < position:
                    total_profit += chosen_vt[vb]
                    profit_position[valid_sellpoint[bin] + lag + lag2 - 1] += chosen_vt[vb]
                    deferred_positions.append((bin, vb + lag))
                    clear_list.append(valid_sellpoint[bin] + lag + lag2 - 1 - vb - lag)
                    # print(deferred_positions)

            else:
                if positions_dict[bin] < position:
                    total_profit += chosen_vt[vb]
                    profit_position[valid_sellpoint[bin + 1] + lag + lag2 - 1] += chosen_vt[vb]
                    deferred_positions.append((bin, vb + lag))
                    deferred_positions.append((bin + 1, vb + lag))
                    clear_list.append(valid_sellpoint[bin + 1] + lag + lag2 - 1 - vb - lag)

        else:
            print("why larger")
    return total_profit, profit_position, sum(positions_dict)


def calculate_profit_2nd(action, length, lag1, lag2, test_field_2):
    action = action.cpu().detach().numpy()
    print(action.shape)
    dout = 4
    m1, m2 = 2, 4
    returns, profit_position = np.zeros((length,)), np.zeros((length,))
    y0 = np.load(str(args.profit_type) + "dout" + str(dout) + "y0" + "_" + ";".join(test_field_2) + "std" + ".npy")
    if args.type == "sell":
        buy_price_base = np.load(str(args.profit_type) + "dout" + str(dout) + "buy2nd_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        buy_price = buy_price_base * (1 - action[:, 1] / 1e4)
        sell_price_base = np.load(str(args.profit_type) + "dout" + str(dout) + "sell_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        sell_price = sell_price_base * (1 + action[:, 0] / 1e4)

        tps = np.load(
            str(args.profit_type) + "dout" + str(dout) + "tps" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + "train.npy")
        tpb = np.load(
            str(args.profit_type) + "dout" + str(dout) + "tpb2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")

        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]

        colored_valid_buypoint = [(vb, 0) for vb in valid_buypoint]
        colored_valid_sellpoint = [(vs, 1) for vs in valid_sellpoint]
        sorted_full_list = (colored_valid_buypoint + colored_valid_sellpoint).sort(key=take_first)
        position = 0
        position_list = []
        deferred_positions = []
        for vp in sorted_full_list:
            i = 0
            retained_items = []
            for i in range(len(deferred_positions)):
                if vp[0] >= deferred_positions[i][0]:
                    if deferred_positions[i][1] == 1:
                        position_list.append(deferred_positions[i][0] - lag1)
                        position += 1
                    else:
                        print(position, len(position_list))
                        position = 0
                        for j in range(len(position_list)):
                            profit = (sell_price[position_list[j]] - buy_price[
                                deferred_positions[i][0] - lag2] + 0.4 / 1e4 * (
                                                  sell_price[position_list[j]] + buy_price[
                                              deferred_positions[i][0] - lag2])) / args.max_p
                            returns[deferred_positions[i][0] - lag2] += 0.5 * profit
                            returns[position_list[j]] += 0.5 * profit
                            profit_position[deferred_positions[i][0]] += profit
                            del position_list[j]

                else:
                    retained_items.append(deferred_positions[i])
            deferred_positions = retained_items
            if vp[1] == 1:
                if position < args.bin:
                    deferred_positions.append((vp[0] + lag1, 1))
            if vp[1] == 0:
                if position != 0:
                    deferred_positions.append((vp[0] + lag2, 0))
    else:
        buy_price_base = np.load(str(args.profit_type) + "dout" + str(dout) + "buy_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        buy_price = buy_price_base * (1 - action[:, 0] / 1e4)
        print("diff?", buy_price.shape)
        sell_price_base = np.load(str(args.profit_type) + "dout" + str(dout) + "sell2nd_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        sell_price = sell_price_base * (1 + action[:, 1] / 1e4)
        tpb = np.load(
            str(args.profit_type) + "dout" + str(dout) + "tpb" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        tps = np.load(
            str(args.profit_type) + "dout" + str(dout) + "tps2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]
        colored_valid_buypoint = [(vb, 1) for vb in valid_buypoint]
        colored_valid_sellpoint = [(vs, 0) for vs in valid_sellpoint]
        sorted_full_list = (colored_valid_buypoint + colored_valid_sellpoint)
        sorted_full_list.sort(key=lambda x: (x[0], x[1]))
        position = 0
        position_list = []
        deferred_positions = []
        clear_list = []
        for vp in sorted_full_list:
            # print(vp)
            i = 0
            retained_items = []
            for i in range(len(deferred_positions)):
                if vp[0] >= deferred_positions[i][0]:
                    if deferred_positions[i][1] == 1:
                        position_list.append(deferred_positions[i][0] - lag1)
                        position += 1
                    else:
                        # position = 0
                        # for j in range(len(position_list)):
                        #     profit = (sell_price[deferred_positions[i][0]-lag2] - buy_price[position_list[j]] + 0.4/1e4*(buy_price[position_list[j]] + sell_price[deferred_positions[i][0]-lag2]))/args.max_p
                        #     # if positinon_list[j] == 1010076:
                        #         # print("here!!!", profit, sell_price[deferred_positions[i][0]-lag2], deferred_positions[i][0]-lag2, buy_price[position_list[j]], position_list[j])
                        #         # exit(-1)
                        #     # returns[deferred_positions[i][0]-lag2] += profit
                        #     # returns[position_list[j]] += profit
                        #     returns[position_list[j]] += profit
                        #     profit_position[deferred_positions[i][0]] += profit
                        #     clear_list.append(deferred_positions[i][0]-lag2-position_list[j])
                        # position_list = []

                        if position > 0:
                            position -= 1
                            profit = (sell_price[deferred_positions[i][0] - lag2] - buy_price[
                                position_list[0]] + 0.4 / 1e4 * (buy_price[position_list[0]] + sell_price[
                                deferred_positions[i][0] - lag2])) / args.max_p
                            # returns[deferred_positions[i][0]-lag2] += 0.5 * profit
                            returns[position_list[0]] += profit
                            profit_position[deferred_positions[i][0]] += profit
                            clear_list.append(deferred_positions[i][0] - lag2 - position_list[0])
                            del position_list[0]
                else:
                    retained_items.append(deferred_positions[i])
            deferred_positions = retained_items
            if vp[1] == 1:
                if position < args.bin:
                    deferred_positions.append((vp[0] + lag1, 1))
            if vp[1] == 0:
                if position > 0:
                    deferred_positions.append((vp[0] + lag2, 0))
    return torch.from_numpy(returns), profit_position


def differential_sharpe_ratio(R_t, A_tm1, B_tm1, eta=0.01):
    A_delta = R_t - A_tm1
    B_delta = R_t ** 2 - B_tm1
    A_t = A_tm1 + eta * A_delta
    B_t = B_tm1 + eta * B_delta
    nominator = B_tm1 * A_delta - (0.5 * A_tm1 * B_delta)
    denominator = (B_tm1 - A_tm1 ** 2) ** 1.5 + 1e-20
    reward = (nominator / denominator)

    return reward, A_t, B_t


def arr(s):
    return np.around(s, 2)


# @jit
def calculate_profit_3nd(action, action2, length, lag1, lag2, test_field_2, type0=args.type, profit_type=19,
                         max_p=1625.985, gamma=args.gamma, consider_zero=args.consider_zero, taker_buy=True):
    if args.fix_open:
        action = 0 * np.ones((length,))
    if args.fix_close:
        action2 = 4 * np.ones((length,))
    print(gamma, consider_zero)
    dout = 4
    m1, m2 = 2, 4
    returns, profit_position, position_np, position_rt = np.zeros((length,)), np.zeros((length,)), np.zeros(
        (length,)), np.zeros((length,))
    returns2 = np.zeros((length,))
    clear_time = 0
    # p1 = np.load(str(profit_type)+"dout"+str(dout)+"y0"+"_"+";".join(test_field_2)+"std"+".npy")
    std = np.load(str(profit_type) + "dout" + str(dout) + "std" + "_" + ";".join(test_field_2) + "std" + ".npy")

    if args.t2:
        abbr = "2"
    else:
        abbr = ""
    if args.use_true_lag:
        lags_list = np.load(
            str(profit_type) + "dout" + str(dout) + "lag" + "_" + ";".join(test_field_2) + "std" + ".npy")
        lags_list = (lags_list // 100).astype(int)
        lags_list = np.nan_to_num(lags_list)
    else:
        lags_list = np.zeros((length,)).astype(int)
    trades = []
    actual_closepoints = []
    time_index = np.load(str(profit_type) + "dout" + str(dout) + "time" + "_" + ";".join(test_field_2) + "std" + ".npy")
    ap1_tk = arr(
        np.load(str(profit_type) + "dout" + str(dout) + "ap1_tk" + "_" + ";".join(test_field_2) + "std" + ".npy"))
    bp1_tk = arr(
        np.load(str(profit_type) + "dout" + str(dout) + "bp1_tk" + "_" + ";".join(test_field_2) + "std" + ".npy"))

    if type0 == "buy":
        buy_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "buy_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        buy_price = buy_price_base * (1 - action / 1e4)
        sell_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell2nd_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        sell_price = sell_price_base * (1 + action2 / 1e4)
        sell_price_base = sell_price_base * (1 + (args.offset) / 1e4)
        tpb = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tpb" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        tps = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tps2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        valid_buypoint_0 = np.logical_and(tpb <= buy_price_base - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_sellpoint_0 = np.logical_and(tps >= sell_price_base + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]

        colored_valid_buypoint_0 = [(vb, 1) for vb in valid_buypoint_0]
        colored_valid_buypoint_set = set(valid_buypoint.flatten())
        colored_valid_buypoint = [(vb, 1) for vb in valid_buypoint]
        colored_valid_sellpoint_0 = [(vs, 0) for vs in valid_sellpoint_0]
        colored_valid_sellpoint_set = set(valid_sellpoint.flatten())
        colored_valid_sellpoint = [(vs, 0) for vs in valid_sellpoint]
        sorted_full_list = (colored_valid_buypoint + colored_valid_sellpoint)
        time_index = np.load(
            str(profit_type) + "dout" + str(dout) + "time" + "_" + ";".join(test_field_2) + "std" + ".npy")

        sorted_full_list.sort(key=lambda x: (x[0], x[1]))
        position = 0
        position_list = []
        deferred_positions = []
        trading_points = []
        trading_points2 = []
        clear_list = []
        plausible_points = valid_buypoint_0
        plausible_points2 = valid_sellpoint_0
        # print("len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint)", len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint))
        related_closing_points = []
        for vpi in range(len(sorted_full_list)):
            # print(vp)
            vp = sorted_full_list[vpi]
            i = 0
            retained_items = []
            for i in range(len(deferred_positions)):
                if vp[0] >= deferred_positions[i][0]:
                    if deferred_positions[i][1] == 1:
                        position_list.append(deferred_positions[i][0] - deferred_positions[i][2])
                        position += 1
                    else:
                        position = 0
                        for j in range(len(position_list)):
                            profit = (sell_price[deferred_positions[i][0] - deferred_positions[i][2]] - buy_price[
                                position_list[j]] + 0.4 / 1e4 * (buy_price[position_list[j]] + sell_price[
                                deferred_positions[i][0] - deferred_positions[i][2]])) / max_p
                            returns[position_list[j]] += profit
                            returns2[deferred_positions[i][0] - deferred_positions[i][2]] += profit
                            profit_position[deferred_positions[i][0]] += profit
                            clear_list.append(deferred_positions[i][0] - deferred_positions[i][2] - position_list[j])
                        position_list = []
                else:
                    retained_items.append(deferred_positions[i])
            deferred_positions = retained_items
            if vp[1] == 1:
                if position < args.bin and len(deferred_positions) < args.flying_thres:
                    if vp[0] in colored_valid_buypoint_set:
                        deferred_positions.append((vp[0] + 1 + lags_list[vp[0]], 1, 1 + lags_list[vp[0]]))
                        trades.append((vp[0], vp[0] + 1 + lags_list[vp[0]]))

            if vp[1] == 0:
                if position > 0:
                    if vp[0] in colored_valid_sellpoint_set:
                        deferred_positions.append((vp[0] + 1 + lags_list[vp[0]], 0, 1 + lags_list[vp[0]]))
                        actual_closepoints.append(vp[0] + 1 + lags_list[vp[0]])

    if type0 == "sell":
        buy_price_base = arr(np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "buy2nd_price_base" + "_" + ";".join(
                test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy"))
        buy_price = arr(buy_price_base * (1 - action2 / 1e4))
        buy_price_base = arr(buy_price_base * (1 - (args.offset) / 1e4))
        sell_price_base = arr(np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy"))
        sell_price = arr(sell_price_base * (1 + action / 1e4))
        ap1 = sell_price_base
        bp1 = buy_price_base
        tps = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tps" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        tpb = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tpb2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        p1 = np.load("y" + "_" + ";".join(test_field_2) + "_std_0" + ".npy")
        pred_y3 = np.load("y3" + "_" + ";".join(test_field_2) + "_std_0" + ".npy")
        pred_y4 = np.load("y4" + "_" + ";".join(test_field_2) + "_std_0" + ".npy")

        sell_price = np.around(ap1 * np.maximum(
            (1 + 0 / 1e4) * (1 + np.clip(0.6 * pred_y3 / 1e4, 10 / 1e4, 2000 / 1e4)) * (
                        1 + np.clip(2 * p1 / 1e4, 0 / 1e4, 50 / 1e4)), 1), 2)
        # sell_price = np.around(ap1*np.maximum((1+0/1e4)*(1+np.clip((10+pred_y3)/1e4, 10/1e4, 2000/1e4))*(1+np.clip(2*p1/1e4, 0/1e4, 50/1e4)), 1), 2)
        buy_price = np.around(bp1 * np.minimum(
            (1 - 0 / 1e4) * (1 - np.clip(0.6 * pred_y4 / 1e4, 10 / 1e4, 2000 / 1e4)) * (
                        1 + np.clip(2 * p1 / 1e4, -50 / 1e4, 0 / 1e4)), 1), 2)
        # buy_price = np.around(bp1*np.minimum((1-0/1e4)*(1-np.clip((10+pred_y4)/1e4, 10/1e4, 2000/1e4))*(1+np.clip(2*p1/1e4, -50/1e4, 0/1e4)), 1), 2)
        # buy_price_std = np.minimum(buy_price, bp1*(1-10/1e4))
        # sell_price_std = np.maximum(sell_price, ap1*(1+10/1e4))
        buy_price_std = buy_price
        sell_price_std = sell_price

        slot = 200
        valid_buypoint_0 = np.logical_and(tpb <= buy_price_base - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint_std = np.logical_and(tpb <= buy_price_std - 0.01, tpb != 0).nonzero()[0]
        print(len(valid_buypoint_std))

        valid_sellpoint_0 = np.logical_and(tps >= sell_price_base + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint_std = np.logical_and(tps >= sell_price_std + 0.01, tps != 0).nonzero()[0]

        colored_valid_buypoint_0 = [(vb, 0) for vb in valid_buypoint_0]
        colored_valid_buypoint_set = set(valid_buypoint.flatten())
        colored_valid_buypoint = [(vb, 0) for vb in valid_buypoint]
        colored_valid_buypoint_std_set = set(valid_buypoint_std.flatten())
        colored_valid_buypoint_std = [(vb, 0) for vb in valid_buypoint_std]

        colored_valid_sellpoint_0 = [(vs, 1) for vs in valid_sellpoint_0]
        colored_valid_sellpoint_set = set(valid_sellpoint.flatten())
        colored_valid_sellpoint = [(vs, 1) for vs in valid_sellpoint]
        colored_valid_sellpoint_std_set = set(valid_sellpoint_std.flatten())
        colored_valid_sellpoint_std = [(vs, 1) for vs in valid_sellpoint_std]

        colored_valid_buypoint_std_both_set = colored_valid_buypoint_set.union(colored_valid_buypoint_std_set)
        colored_valid_sellpoint_std_both_set = colored_valid_sellpoint_set.union(colored_valid_sellpoint_std_set)

        sorted_full_list = [i for i in colored_valid_buypoint_std_both_set.union(colored_valid_sellpoint_std_both_set)]
        sorted_full_list.sort(key=lambda x: x)
        position = 0
        position_list_buy = []
        position_list_sell = []
        deferred_positions_buy = []
        deferred_positions_sell = []
        trading_points = []
        trading_points2 = []
        clear_list = []
        plausible_points = valid_sellpoint_0
        plausible_points2 = valid_buypoint_0
        std = np.load(str(profit_type) + "dout" + str(dout) + "std" + "_" + ";".join(test_field_2) + "std" + ".npy")
        profit_position_std, profit_position_nostd = np.zeros((length,)), np.zeros((length,))
        num_trades_std = 0
        last_vp = 0
        executed_sell_vps = []
        executed_buy_vps = []
        # print("len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint)", len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint), len(valid_buypoint_std), len(valid_sellpoint_std))
        true_position = 0
        true_position_list = []
        position_vps = []
        starts_l = []
        sub_sell_points = []
        sub_buy_points = []
        sub_trades_s = {}
        sub_trades = []
        points_l = {}
        start = 0
        rage = False
        end = 1
        for vpi in range(0, length):

            vp = vpi
            i = 0
            retained_items_buy = []
            for i in range(len(deferred_positions_buy)):
                if vp >= deferred_positions_buy[i][0]:
                    for j in range(deferred_positions_buy[i][3]):
                        position_list_buy.append((deferred_positions_buy[i][0] - deferred_positions_buy[i][2],
                                                  deferred_positions_buy[i][4], deferred_positions_buy[i][1]))
                else:
                    retained_items_buy.append(deferred_positions_buy[i])
            deferred_positions_buy = retained_items_buy
            retained_items_sell = []
            for i in range(len(deferred_positions_sell)):
                if vp >= deferred_positions_sell[i][0]:
                    for j in range(deferred_positions_sell[i][3]):
                        position_list_sell.append((deferred_positions_sell[i][0] - deferred_positions_sell[i][2],
                                                   deferred_positions_sell[i][4], deferred_positions_sell[i][1]))
                else:
                    retained_items_sell.append(deferred_positions_sell[i])
            deferred_positions_sell = retained_items_sell

            min_len = min(len(position_list_buy), len(position_list_sell))
            for j in range(min_len):
                buyed_item = position_list_buy[j]
                selled_item = position_list_sell[j]
                # if abs(buyed_item[0]-selled_item[0]) < 100:
                buyed_item = position_list_buy[j]
                selled_item = position_list_sell[j]
                profit = (selled_item[1] - buyed_item[1] + 0.4 / 1e4 * (selled_item[1]) + 0.4 / 1e4 * (buyed_item[1]))
                returns[buyed_item[0]] += profit
                returns2[selled_item[0]] += profit
                profit_position[max(selled_item[0] + 1 + lags_list[selled_item[0]],
                                    buyed_item[0] + 1 + lags_list[buyed_item[0]])] += profit
                print(profit, selled_item[1], buyed_item[1])
                if "std" in buyed_item[-1] or "std" in selled_item[-1]:
                    num_trades_std += 1
                    profit_position_std[max(selled_item[0] + 1 + lags_list[selled_item[0]],
                                            buyed_item[0] + 1 + lags_list[buyed_item[0]])] += profit
                else:
                    profit_position_nostd[max(selled_item[0] + 1 + lags_list[selled_item[0]],
                                              buyed_item[0] + 1 + lags_list[buyed_item[0]])] += profit
                clear_list.append(abs(buyed_item[0] - selled_item[0]))
                # print(time_index[int(buyed_item[0])], time_index[int(selled_item[0])])
                # print(clear_list)
                sub_trades.append((selled_item[1], buyed_item[1], time_index[selled_item[0]], time_index[buyed_item[0]],
                                   abs(buyed_item[0] - selled_item[0])))
            position_list_sell = position_list_sell[min_len:]
            position_list_buy = position_list_buy[min_len:]
            if len(position_list_buy) > 0:
                position = len(position_list_buy)
            elif len(position_list_sell) > 0:
                position = -len(position_list_sell)
            else:
                position = 0
            end_point = vpi + 1 if vpi + 1 < length else length
            position_np[vp:end_point] = position
            # if len(deferred_positions_buy) != 0 or len(deferred_positions_sell) != 0:
            #     continue
            buy_position = 1
            sell_position = 1

            # if taker_buy:
            #     if position > 5:
            #         sell_position = 2
            #     if position > 10:
            #         sell_position = 5
            #     if position > 20:
            #         sell_position = 7
            # else:
            #     if position < -5:
            #         buy_position = 2
            #     if position < -10:
            #         buy_position = 5
            #     if position < -20:
            #         buy_position = 7

            if vp > start + slot and rage and position == 0 and true_position == 0:
                rage = False

                points_l[start] = (sub_sell_points, sub_buy_points)
                sub_trades_s[start] = sub_trades
                sub_sell_points = []
                sub_buy_points = []
                sub_trades = []

            if taker_buy:
                # if position < 1:

                # if position < 10 and vp in colored_valid_buypoint_std_set and buy_price_std[vp] < ap1_tk[vp+1] and len(deferred_positions_buy) == 0:
                if position < 10 and vp in colored_valid_buypoint_std_set and buy_price_std[vp] < ap1_tk[vp + 1]:
                    # if position < 10 and vp in colored_valid_buypoint_std_set and len(deferred_positions_buy) == 0:
                    deferred_positions_buy.append(
                        (vp + 1 + lags_list[vp], "buystd", 1 + lags_list[vp], buy_position, buy_price_std[vp]))

                    i = 1
                    if not rage:
                        start = vp
                        starts_l.append(vp)
                        rage = True

                    sub_buy_points.append(vp)
                    executed_buy_vps.append(vp)
                    true_position += i * buy_position
                    true_position_list.append(true_position)
                    position_vps.append(vp + 1 + lags_list[vp])
                    print(position, "buy in margin", time_index[vp], vp, buy_price_std[vp],
                          (bp1[vp] - buy_price_std[vp]) / bp1[vp] * 1e4, 1 + lags_list[vp], true_position)

                if position > 0:
                    sell_price_tmp = ap1[vp] * (1 + np.clip(2 * p1[vp] / 1e4, 0 / 1e4, 50 / 1e4))

                    # if arr(sell_price_tmp) <= arr(tps[vp]-0.01) and tps[vp] != 0 and arr(sell_price_tmp) > arr(bp1_tk[vp+1]) and len(deferred_positions_sell) == 0:
                    if arr(sell_price_tmp) <= arr(tps[vp] - 0.01) and tps[vp] != 0 and arr(sell_price_tmp) > arr(
                            bp1_tk[vp + 1]):
                        # if arr(sell_price_tmp) <= arr(tps[vp]-0.01) and tps[vp] != 0 and len(deferred_positions_sell) == 0:

                        deferred_positions_sell.append(
                            (vp + 1 + lags_list[vp], "sellstd", 1 + lags_list[vp], sell_position, sell_price_tmp))
                        sub_sell_points.append(vp)

                        executed_sell_vps.append(vp)
                        true_position -= sell_position
                        true_position_list.append(true_position)
                        position_vps.append(vp)
                        print(position, "sell out maker", time_index[vp], vp, bp1_tk[vp + 1], 1 + lags_list[vp],
                              true_position)
                # if position < 0:
                #     deferred_positions_buy.append((vp+1, "buystdtaker", 1, -position, ap1_tk[vp]))

            else:
                # if position > -10 and vp in colored_valid_sellpoint_std_set and sell_price_std[vp] > bp1_tk[vp+1] and len(deferred_positions_sell) == 0:
                if position > -10 and vp in colored_valid_sellpoint_std_set and sell_price_std[vp] > bp1_tk[vp + 1]:

                    # if position > -10 and vp in colored_valid_sellpoint_std_set and len(deferred_positions_sell) == 0:

                    if not rage:
                        start = vp
                        starts_l.append(vp)
                        rage = True

                    deferred_positions_sell.append(
                        (vp + 1 + lags_list[vp], "sellstd", 1 + lags_list[vp], sell_position, sell_price_std[vp]))

                    print(position, "sell out margin", time_index[vp], vp, sell_price_std[vp],
                          (sell_price_std[vp] - ap1[vp]) / ap1[vp] * 1e4, 1 + lags_list[vp], true_position)

                    i = 1
                    sub_sell_points.append(vp)
                    executed_sell_vps.append(vp)
                    true_position -= sell_position * i
                    true_position_list.append(true_position)
                    position_vps.append(vp + 1 + lags_list[vp])

                if position < 0:
                    buy_price_tmp = bp1[vp] * (1 + np.clip(2 * p1[vp] / 1e4, -50 / 1e4, 0 / 1e4))
                    # if arr(buy_price_tmp) >= arr(tpb[vp]+0.01) and tpb[vp] != 0 and arr(buy_price_tmp) < arr(ap1_tk[vp+1]) and len(deferred_positions_buy) == 0:
                    if arr(buy_price_tmp) >= arr(tpb[vp] + 0.01) and tpb[vp] != 0 and arr(buy_price_tmp) < arr(
                            ap1_tk[vp + 1]):
                        # if arr(buy_price_tmp) >= arr(tpb[vp]+0.01) and tpb[vp] != 0 and len(deferred_positions_buy) == 0:

                        sub_buy_points.append(vp)
                        deferred_positions_buy.append(
                            (vp + 1 + lags_list[vp], "buystd", 1 + lags_list[vp], buy_position, buy_price_tmp))
                        executed_buy_vps.append(vp)
                        true_position += buy_position
                        true_position_list.append(true_position)
                        position_vps.append(vp)
                        print(position, "buy in maker", time_index[vp], vp, ap1_tk[vp + 1], 1 + lags_list[vp],
                              true_position)

            last_vp = vp
            position_rt[vp] = true_position
    if len(clear_list) > 0:
        print("clear time", sum(clear_list) / len(clear_list), len(clear_list))
        clear_time = sum(clear_list) / len(clear_list)
    print("std > 5 trades", num_trades_std, np.sum(profit_position))

    fig, ax1 = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(12)
    ax1.plot(time_index[position_vps], true_position_list, label="rl")
    plt.legend()
    plt.show()
    plt.savefig("results_new_2/result_" + args.arch + ";".join(args.test) + "RLfix" + args.type + "true_position" + str(
        args.bin) + args.rand_number + str(lag1) + str(lag2) + str(taker_buy) + "t2.png")
    plt.close('all')

    fig, ax1 = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(12)
    ax1.plot(time_index, np.cumsum(profit_position), label="rl")
    plt.legend()
    plt.show()
    plt.savefig("results_new_2/result_" + args.arch + ";".join(args.test) + "RLfix" + args.type + "position" + str(
        args.bin) + args.rand_number + str(lag1) + str(lag2) + str(taker_buy) + "t2.png")
    plt.close('all')

    if args.generate_test_predict and args.write_db:

        time_index = np.load(str(args.profit_type) + "dout4time" + "_" + ";".join(args.test) + "std" + ".npy")
        import pandas as pd
        from influxdb import DataFrameClient
        client = DataFrameClient(args.ip, ** **, '**', '********', 'signal')
        data_1 = pd.DataFrame()
        b_ = 4385.71 if args.type == "buy" else 8663
        itemp = '22rl' + args.type + 'sell_filled_opt5'
        data_1[itemp] = pd.DataFrame(sell_price_std[executed_sell_vps])
        data_1['time'] = pd.DataFrame(time_index[executed_sell_vps])
        data_1.index = data_1.time

        data_2 = pd.DataFrame()
        b_ = 4385.71 if args.type == "buy" else 8663
        itemp2 = '22rl' + args.type + 'buy_filled_opt5'
        # data_2[itemp2]=pd.DataFrame(ap1_tk[executed_buy_vps])
        data_2[itemp2] = pd.DataFrame(bp1[executed_buy_vps])
        data_2['time'] = pd.DataFrame(time_index[executed_buy_vps])
        data_2.index = data_2.time

        data_3 = pd.DataFrame()
        b_ = 4385.71 if args.type == "buy" else 8663
        itemp3 = '22rl' + args.type + 'position_true4'
        data_3[itemp3] = pd.DataFrame(true_position_list)
        data_3['time'] = pd.DataFrame(time_index[position_vps])
        data_3.index = data_3.time

        i = 0
        for i in range(1, int(len(data_1[[itemp]]) // 1e5)):
            print((i - 1) * int(1e5), i * int(1e5))
            client.write_points(data_1[[itemp]][(i - 1) * int(1e5):i * int(1e5)], itemp, {})

        print(i * int(1e5), len(data_1[[itemp]]))
        client.write_points(data_1[[itemp]][i * int(1e5):], itemp, {})

        i = 0
        for i in range(1, int(len(data_2[[itemp2]]) // 1e5)):
            print((i - 1) * int(1e5), i * int(1e5))
            client.write_points(data_2[[itemp2]][(i - 1) * int(1e5):i * int(1e5)], itemp2, {})

        print(i * int(1e5), len(data_2[[itemp2]]))
        client.write_points(data_2[[itemp2]][i * int(1e5):], itemp2, {})

        i = 0
        for i in range(1, int(len(data_3[[itemp3]]) // 1e5)):
            print((i - 1) * int(1e5), i * int(1e5))
            client.write_points(data_3[[itemp3]][(i - 1) * int(1e5):i * int(1e5)], itemp3, {})

        print(i * int(1e5), len(data_3[[itemp3]]))
        client.write_points(data_3[[itemp3]][i * int(1e5):], itemp3, {})

    return torch.from_numpy(returns), torch.from_numpy(
        returns2), profit_position, plausible_points, plausible_points2, clear_time, position_np, buy_price, sell_price, profit_position_std, profit_position_nostd


def calculate_profit_13nd(action, action2, length, lag1, lag2, test_field_2, type0=args.type, profit_type=19,
                          max_p=1625.985, gamma=args.gamma, consider_zero=args.consider_zero):
    action = 0 * np.ones((length,))
    action2 = 0 * np.ones((length,))
    # action2 = 4*np.ones((length, ))

    # print(action, action2)
    print(gamma, consider_zero)
    dout = 4
    m1, m2 = 2, 4
    returns, profit_position, position_np = np.zeros((length,)), np.zeros((length,)), np.zeros((length,))
    returns2 = np.zeros((length,))
    clear_time = 0
    p1 = np.load(str(profit_type) + "dout" + str(dout) + "y0" + "_" + ";".join(test_field_2) + "std" + ".npy")
    std = np.load(str(profit_type) + "dout" + str(dout) + "std" + "_" + ";".join(test_field_2) + "std" + ".npy")

    if args.t2:
        abbr = "2"
    else:
        abbr = ""
    if args.use_true_lag:
        lags_list = np.load(
            str(profit_type) + "dout" + str(dout) + "lag" + "_" + ";".join(test_field_2) + "std" + ".npy")
        lags_list = (lags_list // 100).astype(int)
        lags_list = np.nan_to_num(lags_list)
    else:
        lags_list = np.zeros((length,)).astype(int)
    trades = []
    actual_closepoints = []

    if type0 == "buy":
        buy_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "buy_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        buy_price = buy_price_base * (1 - action / 1e4)
        sell_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell2nd_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        sell_price = sell_price_base * (1 + action2 / 1e4)
        sell_price_base = sell_price_base * (1 + (args.offset) / 1e4)
        tpb = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tpb" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        tps = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tps2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        valid_buypoint_0 = np.logical_and(tpb <= buy_price_base - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_sellpoint_0 = np.logical_and(tps >= sell_price_base + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]

        colored_valid_buypoint_0 = [(vb, 1) for vb in valid_buypoint_0]
        colored_valid_buypoint_set = set(valid_buypoint.flatten())
        colored_valid_buypoint = [(vb, 1) for vb in valid_buypoint]
        colored_valid_sellpoint_0 = [(vs, 0) for vs in valid_sellpoint_0]
        colored_valid_sellpoint_set = set(valid_sellpoint.flatten())
        colored_valid_sellpoint = [(vs, 0) for vs in valid_sellpoint]
        sorted_full_list = (colored_valid_buypoint + colored_valid_sellpoint)
        time_index = np.load(
            str(profit_type) + "dout" + str(dout) + "time" + "_" + ";".join(test_field_2) + "std" + ".npy")

        sorted_full_list.sort(key=lambda x: (x[0], x[1]))
        position = 0
        position_list = []
        deferred_positions = []
        trading_points = []
        trading_points2 = []
        clear_list = []
        plausible_points = valid_buypoint_0
        plausible_points2 = valid_sellpoint_0
        # print("len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint)", len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint))
        related_closing_points = []
        for vpi in range(len(sorted_full_list)):
            # print(vp)
            vp = sorted_full_list[vpi]
            i = 0
            retained_items = []
            for i in range(len(deferred_positions)):
                if vp[0] >= deferred_positions[i][0]:
                    if deferred_positions[i][1] == 1:
                        position_list.append(deferred_positions[i][0] - deferred_positions[i][2])
                        position += 1
                    else:
                        position = 0
                        for j in range(len(position_list)):
                            profit = (sell_price[deferred_positions[i][0] - deferred_positions[i][2]] - buy_price[
                                position_list[j]] + 0.4 / 1e4 * (buy_price[position_list[j]] + sell_price[
                                deferred_positions[i][0] - deferred_positions[i][2]])) / max_p
                            returns[position_list[j]] += profit
                            returns2[deferred_positions[i][0] - deferred_positions[i][2]] += profit
                            profit_position[deferred_positions[i][0]] += profit
                            clear_list.append(deferred_positions[i][0] - deferred_positions[i][2] - position_list[j])
                        position_list = []
                else:
                    retained_items.append(deferred_positions[i])
            deferred_positions = retained_items
            if vp[1] == 1:
                if position < args.bin and len(deferred_positions) < args.flying_thres:
                    if vp[0] in colored_valid_buypoint_set:
                        deferred_positions.append((vp[0] + 1 + lags_list[vp[0]], 1, 1 + lags_list[vp[0]]))
                        trades.append((vp[0], vp[0] + 1 + lags_list[vp[0]]))

            if vp[1] == 0:
                if position > 0:
                    if vp[0] in colored_valid_sellpoint_set:
                        deferred_positions.append((vp[0] + 1 + lags_list[vp[0]], 0, 1 + lags_list[vp[0]]))
                        actual_closepoints.append(vp[0] + 1 + lags_list[vp[0]])

    if type0 == "sell":
        buy_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "buy2nd_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        buy_price = buy_price_base * (1 - action2 / 1e4)
        buy_price_base = buy_price_base * (1 - (args.offset) / 1e4)
        sell_price_base = np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell_price_base" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        sell_price = sell_price_base * (1 + action / 1e4)
        buy_price_std = np.load(abbr + str(profit_type) + "dout" + str(dout) + "buy_price_basenop" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy") * (
                                    1 + np.clip(3 * p1 / 1e4, -20 / 1e4, 0)) * (1 - 4 / 1e4) * (1 - std / 1e4)
        sell_price_std = np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell_price_basenop" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy") * (1 + np.clip(3 * p1 / 1e4, 0, 20 / 1e4)) * (
                                     1 + 4 / 1e4) * (1 + std / 1e4)
        bp1 = np.load(abbr + str(profit_type) + "dout" + str(dout) + "buy_price_basenop" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        ap1 = np.load(abbr + str(profit_type) + "dout" + str(dout) + "sell_price_basenop" + "_" + ";".join(
            test_field_2) + "std" + "_" + str(lag1) + str(lag2) + ".npy")
        tps = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tps" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")
        tpb = np.load(
            abbr + str(profit_type) + "dout" + str(dout) + "tpb2nd" + "_" + ";".join(test_field_2) + "std" + "_" + str(
                lag1) + str(lag2) + ".npy")

        valid_buypoint_0 = np.logical_and(tpb <= buy_price_base - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint = np.logical_and(tpb <= buy_price - 0.01, tpb != 0).nonzero()[0]
        valid_buypoint_std = np.logical_and(tpb <= buy_price_std - 0.01, tpb != 0).nonzero()[0]
        valid_sellpoint_0 = np.logical_and(tps >= sell_price_base + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint = np.logical_and(tps >= sell_price + 0.01, tps != 0).nonzero()[0]
        valid_sellpoint_std = np.logical_and(tps >= sell_price_std + 0.01, tps != 0).nonzero()[0]

        colored_valid_buypoint_0 = [(vb, 0) for vb in valid_buypoint_0]
        colored_valid_buypoint_set = set(valid_buypoint.flatten())
        colored_valid_buypoint = [(vb, 0) for vb in valid_buypoint]
        colored_valid_buypoint_std_set = set(valid_buypoint_std.flatten())
        colored_valid_buypoint_std = [(vb, 0) for vb in valid_buypoint_std]

        colored_valid_sellpoint_0 = [(vs, 1) for vs in valid_sellpoint_0]
        colored_valid_sellpoint_set = set(valid_sellpoint.flatten())
        colored_valid_sellpoint = [(vs, 1) for vs in valid_sellpoint]
        colored_valid_sellpoint_std_set = set(valid_sellpoint_std.flatten())
        colored_valid_sellpoint_std = [(vs, 1) for vs in valid_sellpoint_std]
        time_index = np.load(str(args.profit_type) + "dout4time" + "_" + ";".join(args.test) + "std" + ".npy")

        colored_valid_buypoint_std_both_set = colored_valid_buypoint_set.union(colored_valid_buypoint_std_set)
        colored_valid_sellpoint_std_both_set = colored_valid_sellpoint_set.union(colored_valid_sellpoint_std_set)

        sorted_full_list = [i for i in colored_valid_buypoint_std_both_set.union(colored_valid_sellpoint_std_both_set)]
        sorted_full_list.sort(key=lambda x: x)
        # print(np.percentile(p1, 85))
        position = 0
        position_list_buy = []
        position_list_sell = []
        deferred_positions_buy = []
        deferred_positions_sell = []
        trading_points = []
        trading_points2 = []
        clear_list = []
        plausible_points = valid_sellpoint_0
        plausible_points2 = valid_buypoint_0
        std = np.load(str(profit_type) + "dout" + str(dout) + "std" + "_" + ";".join(test_field_2) + "std" + ".npy")

        # print("len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint)", len(stock.train_indices), len(sorted_full_list), len(valid_buypoint_0), len(valid_sellpoint_0), len(valid_buypoint), len(valid_sellpoint), len(valid_buypoint_std), len(valid_sellpoint_std))
        out_of_std_high_steps = 0
        executed_sell_vps = []
        executed_buy_vps = []
        for vpi in range(len(sorted_full_list)):
            vp = sorted_full_list[vpi]
            i = 0
            retained_items_buy = []
            for i in range(len(deferred_positions_buy)):
                if vp >= deferred_positions_buy[i][0]:
                    for j in range(deferred_positions_buy[i][3]):
                        position_list_buy.append(
                            (deferred_positions_buy[i][0] - deferred_positions_buy[i][2], deferred_positions_buy[i][4]))
                else:
                    retained_items_buy.append(deferred_positions_buy[i])
            deferred_positions_buy = retained_items_buy
            retained_items_sell = []
            for i in range(len(deferred_positions_sell)):
                if vp >= deferred_positions_sell[i][0]:
                    for j in range(deferred_positions_sell[i][3]):
                        position_list_sell.append((deferred_positions_sell[i][0] - deferred_positions_sell[i][2],
                                                   deferred_positions_sell[i][4]))
                else:
                    retained_items_sell.append(deferred_positions_sell[i])
            deferred_positions_sell = retained_items_sell

            min_len = min(len(position_list_buy), len(position_list_sell))
            for j in range(min_len):
                buyed_item = position_list_buy[j]
                selled_item = position_list_sell[j]
                profit = (selled_item[1] - buyed_item[1] + 0.4 / 1e4 * (selled_item[1] + buyed_item[1])) / max_p
                returns[buyed_item[0]] += profit
                returns2[selled_item[0]] += profit
                profit_position[max(selled_item[0] + 1 + lags_list[selled_item[0]],
                                    buyed_item[0] + 1 + lags_list[buyed_item[0]])] += profit
                clear_list.append(abs(buyed_item[0] - selled_item[0]))
                # print(buyed_item[0], selled_item[0], lags_list[selled_item[0]], lags_list[buyed_item[0]], profit)
            position_list_sell = position_list_sell[min_len:]
            position_list_buy = position_list_buy[min_len:]
            if len(position_list_buy) > 0:
                position = len(position_list_buy)
            elif len(position_list_sell) > 0:
                position = -len(position_list_sell)
            else:
                position = 0
            end_point = sorted_full_list[vpi + 1] if vpi + 1 < len(sorted_full_list) else length
            position_np[vp:end_point] = position
            if position >= 10:
                buy_position = 1
                sell_position = 3
            elif position >= 5:
                buy_position = 1
                sell_position = 3
            elif position > -5:
                buy_position = 1
                sell_position = 1
            elif position > -10:
                buy_position = 3
                sell_position = 1
            else:
                buy_position = 5
                sell_position = 1
            if std[vp] > 5:
                if vp in colored_valid_buypoint_std_set:
                    deferred_positions_buy.append(
                        (vp + 1 + lags_list[vp], "buystd", 1 + lags_list[vp], buy_position, buy_price_std[vp]))
                    executed_buy_vps.append(vp)
                if vp in colored_valid_sellpoint_std_set:
                    deferred_positions_sell.append(
                        (vp + 1 + lags_list[vp], "sellstd", 1 + lags_list[vp], sell_position, sell_price_std[vp]))
                    executed_sell_vps.append(vp)
            else:
                if position < 0:
                    if vp in colored_valid_buypoint_set:
                        deferred_positions_buy.append(
                            (vp + 1 + lags_list[vp], "buy", 1 + lags_list[vp], buy_position, buy_price[vp]))
                        executed_buy_vps.append(vp)

                if position > -1:
                    if vp in colored_valid_sellpoint_set:
                        deferred_positions_sell.append(
                            (vp + 1 + lags_list[vp], "sell", 1 + lags_list[vp], sell_position, sell_price[vp]))

    if len(clear_list) > 0:
        print("clear time", sum(clear_list) / len(clear_list), len(clear_list))
        clear_time = sum(clear_list) / len(clear_list)
    return torch.from_numpy(returns), torch.from_numpy(
        returns2), profit_position, plausible_points, plausible_points2, clear_time, position_np, buy_price, sell_price


def freeze_model_weights(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
        if p.grad is not None:
            p.grad = None


def act_model_weights(model):
    for n, p in model.named_parameters():
        p.requires_grad = True


def drawdown(xs):
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    min_ = xs[i]
    max_ = np.max(xs[:i])  # start of period
    return max_ - min_


if args.generate_test_predict:
    print(args.test)
    epoch = 0
    args.rand_number = str(args.seed)
    model = torch.load(
        "results/best_model_RL" + args.arch + "_" + ';'.join(args.train) + "_" + ';'.join(args.valid) + "_" + ';'.join(
            args.test2) + args.rand_number + "lag11" + "epoch" + str(args.start) + ".pt")
    model.eval()
    abbr = "2"
    time_index = np.load(str(args.profit_type) + "dout4time" + "_" + ";".join(args.test) + "std" + ".npy")
    action = torch.zeros(1)
    action_ = torch.zeros(1)

    for lag1, lag2 in [(1, 1)]:
        tmp = transAct(action).cpu().detach().squeeze().numpy()
        tmp2 = transAct(action_, "7").cpu().detach().squeeze().numpy()
        tmp[tmp > 10000] = 10000
        returns, returns2, profit_tr, _, _, ct, position_np, buy_price, sell_price, profit_tr_std, profit_tr_nostd = calculate_profit_3nd(
            tmp, tmp2, stock.x2.size(0), lag1, lag2, args.test)
        returns, returns2, profit_tr, _, _, ct, position_np, buy_price, sell_price, profit_tr_std, profit_tr_nostd = calculate_profit_3nd(
            tmp, tmp2, stock.x2.size(0), lag1, lag2, args.test, taker_buy=False)

        reward_sum = torch.sum(returns)
        print("for argmax sampler:", reward_sum)
        # print("drawdown:", drawdown(np.cumsum(profit_tr)))
        lines_position = []
        # y0 = np.load(args.rlbuy_profit_path2+"_"+str(lag1)+str(lag2)+".npy")
        # valid_sellpoint = np.load(args.valid_2ndsellpoint2+"_"+str(lag1)+str(lag2)+".npy")
        # std = np.load(str(args.profit_type)+"dout"+str(4)+"std"+"_"+";".join(args.test)+"std"+".npy")

        for pos in [args.pos]:
            for j in range(4, 4):
                p, profit_fix_position, pos_dict = calculate_profit_with_chosen(valid_sellpoint, y0[:, j],
                                                                                position=args.bin, lag=lag1, lag2=lag2)
                # _, _, profit_fix_position, _, _, _, _, _, _, _, _ = calculate_profit_3nd(j*np.ones((stock.x2.size(0), )), 4*np.ones((stock.x2.size(0), )), stock.x2.size(0), lag1, lag2, args.test)
                lines_position.append(profit_fix_position)
        # buy_price[buy_price>10000] = 10000
        # sell_price[sell_price>10000] = 10000
        if args.use_true_lag:
            lag_or_not = "2"
        else:
            lag_or_not = ""
        if args.flying_thres == 1000:
            flying_thres_or_not = ""
        else:
            flying_thres_or_not = "2"
        if args.write_db and lag1 == 1 and lag2 == 1:
            time_index = np.load(str(args.profit_type) + "dout4time" + "_" + ";".join(args.test) + "std" + ".npy")
            import pandas as pd
            from influxdb import DataFrameClient

            client = DataFrameClient(args.ip, ** **, '**', '********', 'signal')
            data_1 = pd.DataFrame()
            b_ = 4385.71 if args.type == "buy" else 8663
            itemp = flying_thres_or_not + lag_or_not + abbr + '2rl' + args.type + 'profit_lag' + str(lag1) + str(lag2)
            itemp2 = flying_thres_or_not + lag_or_not + abbr + '2rl' + args.type + 'position_lag' + str(lag1) + str(
                lag2)
            itemp3 = abbr + '2rl' + args.type + 'buy price' + str(lag1) + str(lag2)
            itemp4 = abbr + '2rl' + args.type + 'sell price' + str(lag1) + str(lag2)
            itemp5 = flying_thres_or_not + lag_or_not + abbr + '2rl' + args.type + 'position5_lag' + str(lag1) + str(
                lag2)
            itemp6 = flying_thres_or_not + lag_or_not + abbr + '2rl' + args.type + 'profit5_lag' + str(lag1) + str(lag2)

            data_1[itemp2] = pd.DataFrame(position_np * 0.01)
            data_1[itemp3] = pd.DataFrame(buy_price)
            data_1[itemp4] = pd.DataFrame(sell_price)
            data_1[itemp5] = pd.DataFrame(position_np * 5)

            data_1[itemp] = pd.DataFrame(np.cumsum(profit_tr).reshape(profit_tr.shape[0], 1) * args.max_p * 0.01 + b_)
            data_1[itemp6] = pd.DataFrame(np.cumsum(profit_tr).reshape(profit_tr.shape[0], 1) * args.max_p * 5 + b_)

            data_1['time'] = pd.DataFrame(time_index)
            data_1.index = data_1.time
            i = 0
            for i in range(1, int(len(data_1[[itemp]]) // 1e5)):
                print((i - 1) * int(1e5), i * int(1e5))
                client.write_points(data_1[[itemp]][(i - 1) * int(1e5):i * int(1e5)], itemp, {})
                client.write_points(data_1[[itemp2]][(i - 1) * int(1e5):i * int(1e5)], itemp2, {})
                client.write_points(data_1[[itemp3]][(i - 1) * int(1e5):i * int(1e5)], itemp3, {})
                client.write_points(data_1[[itemp4]][(i - 1) * int(1e5):i * int(1e5)], itemp4, {})
                client.write_points(data_1[[itemp5]][(i - 1) * int(1e5):i * int(1e5)], itemp5, {})
                client.write_points(data_1[[itemp6]][(i - 1) * int(1e5):i * int(1e5)], itemp6, {})

            print(i * int(1e5), len(data_1[[itemp]]))
            client.write_points(data_1[[itemp]][i * int(1e5):], itemp, {})
            client.write_points(data_1[[itemp2]][i * int(1e5):], itemp2, {})
            client.write_points(data_1[[itemp3]][i * int(1e5):], itemp3, {})
            client.write_points(data_1[[itemp4]][i * int(1e5):], itemp4, {})
            client.write_points(data_1[[itemp5]][i * int(1e5):], itemp5, {})
            client.write_points(data_1[[itemp6]][i * int(1e5):], itemp6, {})

        exit(-1)


@jit
def calculate_profit_with_chosen_3(valid_sellpoint, chosen_value_total, position=args.bin, lag=1, lag2=1):
    # print(valid_sellpoint)
    chosen_vt = chosen_value_total
    positions_dict = [0 for i in range(len(valid_sellpoint))]
    # profit_positions = [0 for i in range(len(valid_sellpoint))]
    # addedvb_positions = [[] for i in range(len(valid_sellpoint))]

    valid_buypoint = (chosen_vt != 0).nonzero()[0]
    # for i in range(len(chosen_vt)):
    # print(chosen_vt[i])
    profit_position = np.zeros(len(chosen_vt))

    def get_loc(vb):
        for i in range(len(valid_sellpoint)):
            if vb < valid_sellpoint[i]:
                return i, False
            if valid_sellpoint[i] <= vb and vb < valid_sellpoint[i] + lag + lag2 - 1:
                return i, True

    total_profit = 0
    # print(len(valid_buypoint))
    deferred_positions = [(0, 0)]
    deferred_positions = deferred_positions[1:]
    clear_list = []
    # print(len(valid_buypoint), len(valid_sellpoint))
    for vb in valid_buypoint:
        # print(vb)
        # print(total_profit, np.sum(profit_position))
        # if len(clear_list) != 0:
        # print(sum(clear_list)/len(clear_list), len(clear_list), len(valid_buypoint), len(valid_sellpoint))
        i = 0
        retained_items = []
        for i in range(len(deferred_positions)):
            if vb >= deferred_positions[i][1]:
                positions_dict[deferred_positions[i][0]] += 1
            else:
                retained_items.append(deferred_positions[i])
        deferred_positions = retained_items
        # print(positions_dict[:10], max(positions_dict), deferred_positions, lag)
        if vb < valid_sellpoint[-1]:
            bin, atpoint = get_loc(vb)
            if not atpoint:
                if positions_dict[bin] < position:
                    total_profit += chosen_vt[vb]
                    profit_position[valid_sellpoint[bin] + lag + lag2 - 1] += chosen_vt[vb]
                    deferred_positions.append((bin, vb + lag))
                    clear_list.append(valid_sellpoint[bin] + lag + lag2 - 1 - vb - lag)
                    # print(deferred_positions)

            else:
                if positions_dict[bin] < position:
                    total_profit += chosen_vt[vb]
                    profit_position[valid_sellpoint[bin + 1] + lag + lag2 - 1] += chosen_vt[vb]
                    deferred_positions.append((bin, vb + lag))
                    deferred_positions.append((bin + 1, vb + lag))
                    clear_list.append(valid_sellpoint[bin + 1] + lag + lag2 - 1 - vb - lag)

                    # print(deferred_positions)
        else:
            print("why larger")
    return total_profit, profit_position, sum(positions_dict)


max_p = -1 * np.ones((4,))
pathwise_p = -1 * np.ones((epochs, 4))
max_p_sum = 0
max_p_itr_all = 0
stabalize_bn(model, stock.StockAggregate38("train", test_field=args.train), test_field=args.train, argmax=True)
args.rand_number = str(args.seed)
model = torch.load(
    "results/best_model_RL" + args.arch + "_" + ';'.join(args.train) + "_" + ';'.join(args.valid) + "_" + ';'.join(
        args.test2) + args.rand_number + "lag11" + "epoch" + str(args.start) + ".pt")

for epoch in range(epochs):
    model.eval()
    if epoch >= 0:
        _, action, action_, _, _ = calculate_train_RLd3_35(model,
                                                           stock.StockAggregate38("train", test_field=args.train),
                                                           test_field="train", argmax=True)
        if ";".join(args.test) != ";".join(args.train):
            _, action2, action2_, _, _ = calculate_train_RLd3_35(model,
                                                                 stock.StockAggregate38("test", test_field=args.test),
                                                                 test_field="test", argmax=True)
            _, action3, action3_, _, _ = calculate_train_RLd3_35(model,
                                                                 stock.StockAggregate38("test2", test_field=args.test2),
                                                                 test_field="test2", argmax=True)
        # _, action4, action4_, _, _ = calculate_train_RLd3_35(model, stock.StockAggregate38("test4", test_field=args.test4), test_field=args.test4)
        # # for lag1, lag2 in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        for lag1, lag2 in [(1, 1)]:
            tmp = transAct(action).cpu().detach().squeeze().numpy()
            tmp2 = transAct(action_, "7").cpu().detach().squeeze().numpy()
            returns, returns2, profit_tr, _, _, ct, _, _, _ = calculate_profit_3nd(tmp, tmp2, stock.x.size(0), lag1,
                                                                                   lag2, args.train)
            reward_sum = torch.sum(returns)
            print("for argmax sampler:", reward_sum)
            print("drawdown:", drawdown(np.cumsum(profit_tr)))

            wandb.log({"train/reward_sum": reward_sum}, step=epoch)
            wandb.log({"train/close_sharpe": (returns2.mean() / returns2.std()).item()}, step=epoch)
            wandb.log({"train/open_sharpe": (returns.mean() / returns.std()).item()}, step=epoch)

            wandb.log({"train/clear_time": ct}, step=epoch)
            wandb.log({"train/drawdown": drawdown(np.cumsum(profit_tr))}, step=epoch)
            if ";".join(args.test) != ";".join(args.train):
                tmp = transAct(action2).cpu().detach().squeeze().numpy()
                tmp2 = transAct(action2_, "7").cpu().detach().squeeze().numpy()
                returns, returns2, profit0, _, _, ct0, _, _, _ = calculate_profit_3nd(tmp, tmp2, stock.x2.size(0), lag1,
                                                                                      lag2, args.test)
                reward_sum0 = torch.sum(returns)
                print("for argmax sampler:", reward_sum0)
                wandb.log({"test/reward_sum": reward_sum0}, step=epoch)
                wandb.log({"test/clear_time": ct0}, step=epoch)
                wandb.log({"test/close_sharpe": (returns2.mean() / returns2.std()).item()}, step=epoch)
                wandb.log({"test/open_sharpe": (returns.mean() / returns.std()).item()}, step=epoch)
                wandb.log({"test/drawdown": drawdown(np.cumsum(profit0))}, step=epoch)

                tmp = transAct(action3).cpu().detach().squeeze().numpy()
                tmp2 = transAct(action3_, "7").cpu().detach().squeeze().numpy()
                returns, returns2, profit2, _, _, ct2, _, _, _ = calculate_profit_3nd(tmp, tmp2, stock.x3.size(0), lag1,
                                                                                      lag2, args.test2)
                reward_sum2 = torch.sum(returns)
                print("for argmax sampler:", reward_sum2)
                wandb.log({"test2/reward_sum": reward_sum2}, step=epoch)
                wandb.log({"test2/clear_time": ct2}, step=epoch)
                wandb.log({"test2/close_sharpe": (returns2.mean() / returns2.std()).item()}, step=epoch)
                wandb.log({"test2/open_sharpe": (returns.mean() / returns.std()).item()}, step=epoch)
                wandb.log({"test2/drawdown": drawdown(np.cumsum(profit2))}, step=epoch)

            best_epoch = False
            if ";".join(args.test) != ";".join(args.train):
                if reward_sum2 + reward_sum0 >= max_p[2 * (lag1 - 1) + lag2 - 1]:
                    max_p[2 * (lag1 - 1) + lag2 - 1] = reward_sum2 + reward_sum0
                    # if reward_sum0 >= max_p[2*(lag1-1)+lag2-1]:
                    # max_p[2*(lag1-1)+lag2-1] = reward_sum0+
                    max_p_itr = epoch
                    best_epoch = True
                    torch.save(model, "results/best_model_RL" + args.arch + "_" + ';'.join(args.train) + "_" + ';'.join(
                        args.test) + "_" + ';'.join(args.test2) + args.rand_number + "lag" + str(lag1) + str(
                        lag2) + "epoch" + str(epoch) + ".pt")
                print("max_p_itr, max_p, rand_number", max_p_itr, max_p, args.rand_number)

            # returns, _, profit4, _, _ = calculate_profit_3nd(transAct(action4), transAct(action4_),stock.x4.size(0), lag1, lag2, args.test4)
            # reward_sum = torch.sum(returns)
            # print("for argmax sampler:", reward_sum)
            # if lag1*lag2 == 1:
            wandb.log({"test2+0/reward_sum": reward_sum2 + reward_sum0}, step=epoch)
            # else:
            #     wandb.log({"test4/reward_sum"+str(lag1)+str(lag2): reward_sum}, step=epoch)
            #     wandb.log({"test4+2/reward_sum"+str(lag1)+str(lag2): reward_sum+reward_sum2}, step=epoch)
            #     wandb.log({"test4+2+0/reward_sum"+str(lag1)+str(lag2): reward_sum+reward_sum2+reward_sum0}, step=epoch)
            #     wandb.log({"test2+0/reward_sum"+str(lag1)+str(lag2): reward_sum2+reward_sum0}, step=epoch)

            if best_epoch:
                lines_position0 = []
                y0 = np.load(args.rlbuy_profit_path2 + "_" + str(lag1) + str(lag2) + ".npy")
                valid_sellpoint = np.load(args.valid_2ndsellpoint2 + "_" + str(lag1) + str(lag2) + ".npy")
                for pos in [args.pos]:
                    for j in range(0, 6):
                        # p, profit_fix_position, pos_dict = calculate_profit_with_chosen(valid_sellpoint, y0[:, j], position=args.bin, lag=lag1, lag2=lag2)
                        returns, returns2, profit_fix_position, _, _, ct, position_np, buy_price, sell_price = calculate_profit_3nd(
                            j * np.ones((stock.x2.size(0),)), 4 * np.ones((stock.x2.size(0),)), stock.x2.size(0), lag1,
                            lag2, args.test)
                        print(np.sum(profit_fix_position), torch.sum(returns))
                        lines_position0.append(profit_fix_position)
                fig, ax1 = plt.subplots()
                fig.set_figwidth(16)
                fig.set_figheight(12)
                for j in range(0, 6):
                    ax1.plot(np.cumsum(lines_position0[j])[::interval], label=str(j))
                ax1.plot(np.cumsum(profit0)[::interval], label="rl")
                plt.legend()
                plt.show()
                plt.savefig(
                    "results_new_2/result_" + args.arch + ";".join(args.test) + "RLfix" + args.type + "position" + str(
                        args.bin) + args.rand_number + str(lag1) + str(lag2) + ".png")
                plt.close('all')

                lines_position = []
                y0 = np.load(args.rlbuy_profit_path + "_" + str(lag1) + str(lag2) + ".npy")
                valid_sellpoint = np.load(args.valid_2ndsellpoint + "_" + str(lag1) + str(lag2) + ".npy")
                for pos in [args.pos]:
                    for j in range(0, 6):
                        # p, profit_fix_position, pos_dict = calculate_profit_with_chosen(valid_sellpoint, y0[:, j], position=args.bin, lag=lag1, lag2=lag2)
                        returns, returns2, profit_fix_position, _, _, ct, position_np, buy_price, sell_price = calculate_profit_3nd(
                            j * np.ones((stock.x3.size(0),)), 4 * np.ones((stock.x3.size(0),)), stock.x3.size(0), lag1,
                            lag2, args.test2)
                        lines_position.append(profit_fix_position)
                fig, ax1 = plt.subplots()
                fig.set_figwidth(16)
                fig.set_figheight(12)
                for j in range(0, 6):
                    ax1.plot(np.cumsum(lines_position[j])[::interval], label=str(j))
                ax1.plot(np.cumsum(profit_tr)[::interval], label="rl")
                plt.legend()
                plt.show()
                plt.savefig(
                    "results_new_2/result_" + args.arch + ";".join(args.train) + "RLfix" + args.type + "position" + str(
                        args.bin) + args.rand_number + str(lag1) + str(lag2) + ".png")
                plt.close('all')

                lines_position2 = []
                y0 = np.load(args.rlbuy_profit_path3 + "_" + str(lag1) + str(lag2) + ".npy")
                valid_sellpoint = np.load(args.valid_2ndsellpoint3 + "_" + str(lag1) + str(lag2) + ".npy")
                for pos in [args.pos]:
                    for j in range(0, 6):
                        # p, profit_fix_position, pos_dict = calculate_profit_with_chosen(valid_sellpoint, y0[:, j], position=args.bin, lag=lag1, lag2=lag2)
                        returns, returns2, profit_fix_position, _, _, ct, position_np, buy_price, sell_price = calculate_profit_3nd(
                            j * np.ones((stock.x.size(0),)), 4 * np.ones((stock.x.size(0),)), stock.x.size(0), lag1,
                            lag2, args.train)
                        lines_position2.append(profit_fix_position)
                fig, ax1 = plt.subplots()
                fig.set_figwidth(16)
                fig.set_figheight(12)
                for j in range(0, 6):
                    ax1.plot(np.cumsum(lines_position2[j])[::interval], label=str(j))
                ax1.plot(np.cumsum(profit2)[::interval], label="rl")
                plt.legend()
                plt.show()
                plt.savefig(
                    "results_new_2/result_" + args.arch + ";".join(args.test2) + "RLfix" + args.type + "position" + str(
                        args.bin) + args.rand_number + str(lag1) + str(lag2) + ".png")
                plt.close('all')

            # print("All avg max_p_itr, max_p_sum, rand_number", max_p_itr_all, max_p_sum, args.rand_number)

    action_list, logprobs_list, returns_list, pla_list = [], [], [], []
    action_list2, logprobs_list2, returns_list2, pla_list2 = [], [], [], []
    obs, action, action2, logprobs, logprobs2 = calculate_train_RLd3_35(model, stock.StockAggregate38("train",
                                                                                                      test_field=args.train),
                                                                        test_field="train", r=args.repeat_times)
    print(obs.size(), action.size(), action2.size(), logprobs.size(), logprobs2.size())
    for i in range(args.repeat_times):
        # print("----------",action,"???????", transAct(action[:, :, i]))
        returns, returns2, profit_tr, plausible_points, plausible_points2, _, _, _, _ = calculate_profit_3nd(
            transAct(action[:, :, i]).cpu().detach().squeeze().numpy(),
            transAct(action2[:, i], "7").cpu().detach().squeeze().numpy(), stock.x.size(0), lag1, lag2, args.train)
        action_list.append(action[plausible_points, :, i])
        logprobs_list.append(logprobs[plausible_points, i])
        returns_list.append(returns[plausible_points])
        pla_list.append(torch.from_numpy(plausible_points))
        action_list2.append(action2[plausible_points2, i])
        logprobs_list2.append(logprobs2[plausible_points2, i])
        returns_list2.append(returns2[plausible_points2])
        pla_list2.append(torch.from_numpy(plausible_points2))
        print("rewards, values", returns[plausible_points].mean(), returns2[plausible_points2].mean())

    action = torch.cat(action_list, dim=0)
    logprobs = torch.cat(logprobs_list, dim=0)
    returns = torch.cat(returns_list, dim=0)
    plausible_points = torch.cat(pla_list, dim=0)
    action2 = torch.cat(action_list2, dim=0)
    logprobs2 = torch.cat(logprobs_list2, dim=0)
    returns2 = torch.cat(returns_list2, dim=0)
    plausible_points2 = torch.cat(pla_list2, dim=0)

    print("rewards, values", returns.mean(), returns2.mean())
    returns = (returns / stock.abs_profit - stock.mean_profit) / stock.std_profit
    returns2 = (returns2 / stock.abs_profit2 - stock.mean_profit2) / stock.std_profit2

    train_dataset = stock.StockAggregate42(obs, logprobs, action, returns, plausible_points)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    train_dataset2 = stock.StockAggregate42(obs, logprobs2, action2, returns2, plausible_points2)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                pin_memory=True)
    for g in opt.param_groups:
        g['lr'] = args.lr
    if not args.fix_open:
        train(model, train_loader, opt, epoch, 0)
    div = max(len(train_loader) // len(train_loader2), 1)
    for g in opt.param_groups:
        g['lr'] = args.lr * 0.1

    if not args.fix_close:
        for divi in range(div):
            train(model, train_loader2, opt, epoch, 1)