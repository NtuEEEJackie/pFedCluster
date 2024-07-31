import numpy as np
import torch

def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.eval()

    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                total_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def compute_local_test_accuracy(model, dataloader, data_distribution):
    # 将模型设置为评估模式，不进行梯度计算
    model.eval()

    # 初始化各标签总数和正确预测数的数组
    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    #print(f"[baobao] model:{model} data_distribution:{data_distribution}")
    # 将模型移动到GPU上
    model.cuda()

    # 初始化广义测试的总样本数和正确分类数
    generalized_total, generalized_correct = 0, 0

    # 使用torch.no_grad()上下文管理器，避免计算梯度
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for batch_idx, (x, target) in enumerate(dataloader):
            # 将输入数据和目标标签移动到GPU上
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            # 前向传播获取模型的输出
            out = model(x)

            # 获取模型预测的标签
            _, pred_label = torch.max(out.data, 1)

            # 计算正确预测的标签
            correct_filter = (pred_label == target.data)

            # 更新广义测试的总样本数和正确分类数
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()

            # 更新每个标签的总数和正确预测数
            for i, true_label in enumerate(target.data):
                total_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1

    # 计算个性化测试的总正确预测数和总样本数
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()

    # 将模型切换回CPU
    model.to('cpu')

    # 返回个性化测试准确率和广义测试准确率
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def compute_acc(net, test_data_loader):
    net.cuda()
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)


def compute_acc(net, test_data_loader):
    # 将模型移到GPU上进行计算
    net.cuda()
    # 将模型设置为评估模式，不进行梯度计算
    net.eval()

    # 初始化正确分类数和总样本数
    correct, total = 0, 0

    # 使用torch.no_grad()上下文管理器，避免计算梯度
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for batch_idx, (x, target) in enumerate(test_data_loader):
            # 将输入数据和目标标签移动到GPU上
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            # 前向传播获取模型的输出
            out = net(x)

            # 获取模型预测的标签
            _, pred_label = torch.max(out.data, 1)

            # 更新总样本数
            total += x.data.size()[0]

            # 更新正确分类数
            correct += (pred_label == target.data).sum().item()

    # 将模型切换回CPU
    net.to('cpu')

    # 返回模型在测试集上的准确率
    return correct / float(total)


def evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            val_local_dl = val_local_dls[net_id]
            data_distribution = data_distributions[net_id]

            val_acc = compute_acc(global_model, val_local_dl)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(global_model, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()
