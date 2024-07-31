import torch
import numpy as np
import copy
import cvxpy as cp

from sklearn.cluster import SpectralClustering
import torch.optim as optim

import torch.nn.functional as F

def compute_local_test_accuracy(model, dataloader, data_distribution):
    model.eval()
    toatl_label_num = np.zeros(len(data_distribution))
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
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()

    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def calculate_initial_threshold(similarity_matrix):
    n = similarity_matrix.size(0)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    similarities = similarity_matrix[mask]
    threshold = torch.quantile(similarities, 0.1)
    return threshold.item()



def adjust_similarity_threshold(current_similarity_threshold, performance_metric):
    #current_similarity_threshold = calculate_initial_threshold(similarity_matrix)
    if performance_metric > 0.85:
        current_similarity_threshold = min(current_similarity_threshold + 0.005, 0.9)
    elif performance_metric < 0.8:
        current_similarity_threshold = max(current_similarity_threshold - 0.005, -0.9)
    return current_similarity_threshold

class cosine_difference_array:
    def __init__(self, x, y, cosine_difference):
        self.x = x
        self.y = y
        self.cosine_difference = cosine_difference

def aggregation_dynamic(nets_this_round, initial_global_parameters, dw, similarity_matric, performance_metric):
    length_nets = len(nets_this_round)
    index_clientid = list(nets_this_round.keys())
    cluster_difference = [[client_id] for client_id in index_clientid]
    diff_matric = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    current_similarity_threshold = calculate_initial_threshold(diff_matric)
    adjust_similarity_threshold(current_similarity_threshold, performance_metric)
    print(f"[baojunyan] current_similarity_threshold: {current_similarity_threshold}")

    for i in range(length_nets):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
    while True:
        cosine_difference_list = []
        length_dw = len(dw)
        for i in range(length_dw):
            for j in range(i + 1, length_dw):
                if similarity_matric == "all":
                    diff = -torch.nn.functional.cosine_similarity(
                        weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                        weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                    if diff < -0.9:
                        diff = -1.0
                elif similarity_matric == "fc":
                    diff = - torch.nn.functional.cosine_similarity(
                        weight_flatten(dw[index_clientid[i]]).unsqueeze(0),
                        weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                    if diff < -0.9:
                        diff = -1.0
                cosine_difference_list.append(cosine_difference_array(i, j, diff))

        sorted_cosine_difference_list = sorted(cosine_difference_list, key=lambda x: x.cosine_difference)
        max_similarity_item = sorted_cosine_difference_list[len(cosine_difference_list) - 1]

        if max_similarity_item.cosine_difference < current_similarity_threshold or len(cosine_difference_list) == 1:
            break

        m, n = max_similarity_item.x, max_similarity_item.y
        for key in dw[index_clientid[m]]:
            dw[index_clientid[m]][key] = (dw[index_clientid[m]][key] + dw[index_clientid[n]][key]) / 2
        del dw[index_clientid[n]]
        cluster_difference[m] += cluster_difference[n]
        del cluster_difference[n]
        # for cluster in cluster_difference:
        #     print(f"[baojy] cluster: {cluster}")
    print(f"[baojunyan] cluster_difference: {cluster_difference}")
    # for item in sorted_cosine_difference_list:
    # print(f"[baojunyan] i: {item.x}, j: {item.y}, cosine_difference: {item.cosine_difference}")
    return cluster_difference

#
# def model_distillation(teacher_model, student_model, dataloader, device='cuda'):
#     teacher_model = teacher_model.to(device)
#     student_model = student_model.to(device)
#
#     teacher_model.eval()
#     student_model.train()
#
#     optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
#     distillation_criterion = torch.nn.KLDivLoss(reduction='batchmean')
#
#     for x, target in dataloader:
#         x = x.to(device)
#         target = target.to(device)
#
#         student_output = student_model(x)
#         with torch.no_grad():
#             teacher_output = teacher_model(x)
#
#         loss = distillation_criterion(
#             F.log_softmax(student_output / 2., dim=1),
#             F.softmax(teacher_output / 2., dim=1)
#         )
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     student_model.eval()
#
# def Knowledge_Sharing(nets_this_round, cluster, cluster_difference, train_local_dls, best_test_acc_list_res):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     cluster_accs = np.array([best_test_acc_list_res[i] for i in cluster])
#     max_index_in_cluster_accs = np.argmax(cluster_accs)
#     teacher_model_id = cluster[max_index_in_cluster_accs]
#
#     teacher_model = nets_this_round[teacher_model_id].to(device)
#     for cluster_other in cluster_difference:
#         if cluster_other != cluster:
#             cluster_other_accs = np.array([best_test_acc_list_res[i] for i in cluster_other])
#             max_index_in_cluster_other_accs = np.argmin(cluster_other_accs)
#
#             student_model = nets_this_round[max_index_in_cluster_other_accs].to(device)
#             dataloader = train_local_dls[max_index_in_cluster_other_accs]
#             model_distillation(teacher_model, student_model, dataloader, device)
#
#             student_model.to('cpu')
#     teacher_model.to('cpu')

def Knowledge_Sharing(nets_this_round, cluster, cluster_difference, train_local_dls, best_test_acc_list_res, temperature, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cluster_accs = np.array([best_test_acc_list_res[i] for i in cluster])
    max_index_in_cluster_accs = np.argmax(cluster_accs)
    teacher_model_id = cluster[max_index_in_cluster_accs]

    teacher_model = nets_this_round[teacher_model_id].to(device)
    for cluster_other in cluster_difference:
        if cluster_other != cluster:
            cluster_other_accs = np.array([best_test_acc_list_res[i] for i in cluster_other])
            max_index_in_cluster_other_accs = np.argmax(cluster_other_accs)

            student_model = nets_this_round[cluster_other[max_index_in_cluster_other_accs]].to(device)
            dataloader = train_local_dls[cluster_other[max_index_in_cluster_other_accs]]
            model_distillation(teacher_model, student_model, dataloader, device, temperature, learning_rate)

            student_model.to('cpu')
    teacher_model.to('cpu')

import torch.nn as nn
def model_distillation(teacher_model, student_model, dataloader, device='cuda', temperature=2.0, learning_rate=1e-4):
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    distillation_criterion = nn.KLDivLoss(reduction='batchmean')

    for x, target in dataloader:
        x = x.to(device)
        target = target.to(device)

        student_output = student_model(x)
        with torch.no_grad():
            teacher_output = teacher_model(x)

        loss = distillation_criterion(
            F.log_softmax(student_output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_model.eval()

def lists_are_equal_unordered(M, N):
    if len(M) != len(N):
        return True
    for i in range(len(M)):
        if sorted(M[i]) != sorted(N[i]):
            return True
    return False
#
#
# def aggregation_by_cluster1(cfg, cluster_difference, nets_this_round, global_w, dw, similarity_matric,train_local_dls,best_test_acc_list_res):
#     tmp_client_state_dict = {}
#     diff_matric = cal_model_cosine_difference(nets_this_round, global_w, dw, similarity_matric)
#     #weight_matric = quadratic_planning(diff_matric)
#     cluster_model_vectors = {}
#     for client_id in nets_this_round.keys():
#         tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
#         cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
#         for key in tmp_client_state_dict[client_id]:
#             tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])
#
#     cluster_call_status = {frozenset(cluster): True for cluster in cluster_difference}
#
#     for client_id in nets_this_round.keys():
#         tmp_client_state = tmp_client_state_dict[client_id]
#         cluster_model_state = cluster_model_vectors[client_id]
#         diff_client_id = diff_matric[client_id]
#         for cluster in cluster_difference:
#             if client_id in cluster:
#                 aggregation_weight_vector = cluster
#                 break
#         for cluster_id in aggregation_weight_vector:
#             cluster_para = nets_this_round[cluster_id].state_dict()
#             for key in tmp_client_state:
#                 tmp_client_state[key] += cluster_para[key] * diff_client_id[cluster_id]
#
#         for cluster_id in aggregation_weight_vector:
#             cluster_para = weight_flatten_all(nets_this_round[cluster_id].state_dict())
#             cluster_model_state += cluster_para * (diff_client_id[cluster_id] / torch.linalg.norm(cluster_para))
#
#     for client_id in nets_this_round.keys():
#         nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])
#     #print(f"[baojy2323232] cluster_model_vectors:{cluster_model_vectors}")
#
#     return cluster_model_vectors


import torch
import copy
import numpy as np
import cvxpy as cp

def optimizing_graph_matrix_neighbor(graph_matrix, cluster, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array([fed_avg_freqs[i] for i in cluster])
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()

        graph_matrix[cluster[i], cluster] = torch.Tensor(x.value)
    return graph_matrix


def update_graph_matrix_neighbor(nets_this_round, cluster, initial_global_parameters, dw, fed_avg_freqs, lambda_1,
                                 similarity_matric):

    model_difference_matrix = cal_model_cosine_difference1(nets_this_round, initial_global_parameters, dw,
                                                          similarity_matric, cluster)

    graph_matrix = torch.ones(len(nets_this_round), len(nets_this_round)) / (len(nets_this_round) - 1)  # Collaboration Graph
    graph_matrix[range(len(nets_this_round)), range(len(nets_this_round))] = 0
    subgraph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, cluster, model_difference_matrix, lambda_1,
                                                    fed_avg_freqs)
    # print(f'Model difference: {model_difference_matrix[0]}')
    return subgraph_matrix


def aggregation_by_cluster(cfg, cluster_difference, nets_this_round, global_w, dw, similarity_matric, fed_avg_freqs, train_local_dls, best_test_acc_list_res,alpha):

    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    #diff_matric = cal_model_cosine_difference(nets_this_round, global_w, dw, similarity_matric)

    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)

        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))

        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])
    cluster_call_status = {frozenset(cluster): True for cluster in cluster_difference}


    for cluster in cluster_difference:

        optimal_weights = update_graph_matrix_neighbor(nets_this_round, cluster, global_w, dw, fed_avg_freqs, 0.8, similarity_matric)

        for client_id in cluster:
            tmp_client_state = tmp_client_state_dict[client_id]
            cluster_model_state = cluster_model_vectors[client_id]
            weight_vector = optimal_weights[client_id]


            for client_id in cluster:
                cluster_para = nets_this_round[client_id].state_dict()

                for key in tmp_client_state:
                    tmp_client_state[key] += cluster_para[key] * weight_vector[client_id]

            for client_id in cluster:
                net_para = weight_flatten_all(nets_this_round[client_id].state_dict())
                cluster_model_state += net_para * (weight_vector[client_id] / torch.linalg.norm(net_para))
        cluster_key = frozenset(cluster)
        if cluster_call_status[cluster_key]:

            Knowledge_Sharing(nets_this_round, cluster, cluster_difference, train_local_dls,
                              best_test_acc_list_res, temperature=1.0, learning_rate=1e-5)

            cluster_call_status[cluster_key] = False
    # for client_id in nets_this_round.keys():
    #     tmp_client_state = tmp_client_state_dict[client_id]
    #     global_state = global_w
    #     for key in tmp_client_state:
    #         tmp_client_state[key] = alpha * tmp_client_state[key] + (1 - alpha) * global_state[key]
    #     nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

    return cluster_model_vectors


def quadratic_planning(model_difference_matrix):
    n = model_difference_matrix.shape[0]
    graph_matrix = cp.Variable((n, n))
    objective = cp.Minimize(cp.norm(graph_matrix - (-model_difference_matrix), 'fro'))

    constraints = [
        cp.diag(graph_matrix) == 0,
        graph_matrix >= 0,
        cp.sum(graph_matrix, axis=1) == 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    optimized_matrix = torch.Tensor(graph_matrix.value)
    return torch.Tensor(optimized_matrix)



def cal_model_cosine_difference1(nets_this_round, initial_global_parameters, dw, similarity_matric, cluster):
    model_similarity_matrix = torch.zeros((len(cluster), len(cluster)))

    for i in range(len(cluster)):
        model_i = nets_this_round[cluster[i]].state_dict()
        for key in dw[cluster[i]]:
            dw[cluster[i]][key] = model_i[key] - initial_global_parameters[key]

    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[cluster[i]]).unsqueeze(0),
                                                               weight_flatten_all(dw[cluster[j]]).unsqueeze(0))
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[cluster[i]]).unsqueeze(0),
                                                               weight_flatten(dw[cluster[j]]).unsqueeze(0))
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix
def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))

    index_clientid = list(nets_this_round.keys())

    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]

    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if i == j:
                model_similarity_matrix[i, j] = 1
                continue
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    if np.isnan(model_similarity_matrix).any():
        model_similarity_matrix = np.nan_to_num(model_similarity_matrix)
    #print(f"[baojunyan] model_similarity_matrix: {model_similarity_matrix}")
    return model_similarity_matrix
#
# import torch
#
# def cal_model_cosine_difference2(nets_this_round, initial_global_parameters, dw, similarity_matric, performance_weights, data_importance_weights):
#     combined_weights = {}
#     for client_id in nets_this_round.keys():
#         combined_weights[client_id] = performance_weights[client_id] * 0.5 + data_importance_weights[client_id] * 0.5  # 可以调整权重比例
#
#     model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))
#     index_clientid = list(nets_this_round.keys())
#
#     for i in range(len(nets_this_round)):
#         model_i = nets_this_round[index_clientid[i]].state_dict()
#         for key in dw[index_clientid[i]]:
#             dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
#
#     for i in range(len(nets_this_round)):
#         for j in range(i, len(nets_this_round)):
#             if similarity_matric == "all":
#                 diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
#                 diff *= combined_weights[index_clientid[i]] * combined_weights[index_clientid[j]]  # 应用加权机制
#                 if diff < -0.9:
#                     diff = -1.0
#                 model_similarity_matrix[i, j] = diff
#                 model_similarity_matrix[j, i] = diff
#             elif similarity_matric == "fc":
#                 diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0), weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
#                 diff *= combined_weights[index_clientid[i]] * combined_weights[index_clientid[j]]  # 应用加权机制
#                 if diff < -0.9:
#                     diff = -1.0
#                 model_similarity_matrix[i, j] = diff
#                 model_similarity_matrix[j, i] = diff
#
#     return model_similarity_matrix



def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)


def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)

def generate_sparse_mask(shape, sparsity_level):
    total_elements = np.prod(shape)
    non_sparse_elements = int(total_elements * sparsity_level)
    mask_flat = np.concatenate([np.ones(non_sparse_elements), np.zeros(total_elements - non_sparse_elements)])
    np.random.shuffle(mask_flat)
    return torch.from_numpy(mask_flat).float().reshape(shape)

def apply_sparse_mask_to_model(model, sparsity_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = generate_sparse_mask(param.shape, sparsity_level).to(param.device)
            param.data = param.data * mask


def adjust_sparsity_level(current_round, total_rounds):
    base_sparsity = 0.8
    max_increase = 0.9

    if current_round <= total_rounds / 2:
        sparsity_level = base_sparsity + (max_increase / 2) * (current_round / (total_rounds / 2))
    else:
        sparsity_level = base_sparsity + (max_increase / 2) + (
                (max_increase / 2) * ((current_round - (total_rounds / 2)) / (total_rounds / 2)))

    sparsity_level = max(0, min(sparsity_level, 1))
    return sparsity_level


def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def merge_dataloaders(client_ids, train_local_dls):
    merged_dataset = torch.utils.data.ConcatDataset([train_local_dls[client_id].dataset for client_id in client_ids])
    merged_dataloader = torch.utils.data.DataLoader(merged_dataset, batch_size=64, shuffle=True)
    return merged_dataloader


def aggregate_models(client_ids, nets_this_round):
    aggregated_model = copy.deepcopy(nets_this_round[client_ids[0]])

    for name in aggregated_model.state_dict():
        avg_param = torch.mean(torch.stack([nets_this_round[client_id].state_dict()[name] for client_id in client_ids]),
                               dim=0)
        aggregated_model.state_dict()[name].copy_(avg_param)

    return aggregated_model

#
# def adjust_sparsity_level(current_round, total_rounds, performance_drop=False):
#     base_sparsity = 0.1
#     max_increase = 0.5
#     if performance_drop:
#         max_increase *= 0.5
#     sparsity_level = base_sparsity + (max_increase * (current_round / total_rounds))
#     return min(0.9, sparsity_level)
