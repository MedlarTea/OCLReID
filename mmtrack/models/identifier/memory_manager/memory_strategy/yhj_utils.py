import torch
from mmtrack.models.identifier.utils.utils import maybe_cuda, mini_batch_deep_features, euclidean_distance, nonzero_indices, ohe_label
# from utils.setup_elements import n_classes
from .buffer_utils import ClassBalancedRandomSampling, random_retrieve, get_grad_vector
import copy

from mmtrack.models.identifier.track_center.tracklet import Tracklet
# from .buffer import Buffer
import numpy as np
import random

from .buffer_utils import ClassBalancedRandomSampling
from collections import defaultdict
import torch.nn.functional as F

def random_retrieve(buffer, num_retrieve, excl_indices=None, return_indices=False):
    filled_indices = [k for i in buffer.buffer_tracker.class_index_cache.values() for k in i]
    # print("filled_indices: ", filled_indices)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    num_retrieve = min(num_retrieve, valid_indices.shape[0])
    indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()

    x = buffer.buffer_img[indices]
    y = buffer.buffer_label[indices]

    if return_indices:
        return x, y, indices
    else:
        return x, y
        """Retrieve images and vis_map for lt-classifier training

        Output:
            imgs: for training with shape of (B, 3, H, W)
            labels: labels with shape of (B)
            vis_indicator: for visibility indication with shape of (B, 10)---front back 4 parts and 1 global
        """
        indices = []
        imgs = []
        labels = []
        vis_indicators = []

        # [4, 9]
        for i in range(retriever.part_nums):
            part_indices = retriever.full_indexes[i].difference(buffer.buffer_tracker.remaining_indexes[i])
            if len(part_indices) == 0:
                continue
            part_indices = np.array(list(part_indices))
            num_retrieve = min(num_retrieve, part_indices.shape[0])
            part_indices = torch.from_numpy(np.random.choice(part_indices, num_retrieve, replace=False)).long()
            indices.append(part_indices)

            buffer_img = getattr(buffer, "buffer_img_{}".format(i))
            buffer_label = getattr(buffer, "buffer_label_{}".format(i))
            buffer_vis_indicator = getattr(buffer, "buffer_vis_indicator_{}".format(i))

            imgs.append(buffer_img[part_indices])
            labels.append(buffer_label[part_indices])
            vis_indicators.append(buffer_vis_indicator[part_indices])
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        vis_indicators = torch.concat(vis_indicators)

        if return_indices:
            return imgs, labels, vis_indicators, indices
        else:
            return imgs, labels, vis_indicators

def part_random_retrieve(buffer, retriever,num_retrieve, excl_indices=None, return_indices=False):
        """Retrieve images and vis_map for lt-classifier training

        Output:
            imgs: for training with shape of (B, 3, H, W)
            labels: labels with shape of (B)
            vis_indicator: for visibility indication with shape of (B, 10)---front back 4 parts and 1 global
        """
        indices = []
        imgs = []
        labels = []
        vis_maps = []
        vis_indicators = []

        # [4, 9]
        for i in range(retriever.vis_map_nums, retriever.part_nums, retriever.part_nums//2):
            part_indices = retriever.full_indexes[i].difference(buffer.buffer_tracker.remaining_indexes[i])
            if len(part_indices) == 0:
                continue
            part_indices = np.array(list(part_indices))
            num_retrieve = min(num_retrieve, part_indices.shape[0])
            part_indices = torch.from_numpy(np.random.choice(part_indices, num_retrieve, replace=False)).long()
            indices.append(part_indices)

            buffer_img = getattr(buffer, "buffer_img_{}".format(i))
            buffer_label = getattr(buffer, "buffer_label_{}".format(i))
            buffer_vis_map = getattr(buffer, "buffer_vis_map_{}".format(i))
            buffer_vis_indicator = getattr(buffer, "buffer_vis_indicator_{}".format(i))

            imgs.append(buffer_img[part_indices])
            labels.append(buffer_label[part_indices])
            vis_maps.append(buffer_vis_map[part_indices])
            vis_indicators.append(buffer_vis_indicator[part_indices])
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        vis_maps = torch.concat(vis_maps)
        vis_indicators = torch.concat(vis_indicators)

        if return_indices:
            return imgs, labels, vis_maps, vis_indicators, indices
        else:
            return imgs, labels, vis_maps, vis_indicators

def random_retrieve_features(buffer, num_retrieve, excl_indices=None, return_indices=False):
    filled_indices = [k for i in buffer.buffer_tracker.class_index_cache.values() for k in i]
    # print("filled_indices: ", filled_indices)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    num_retrieve = min(num_retrieve, valid_indices.shape[0])
    indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()

    x = buffer.buffer_feature[indices, :buffer.params.deep_feature_dim]
    y = buffer.buffer_label[indices]

    if return_indices:
        return x, y, indices
    else:
        return x, y

def part_random_retrieve_features(buffer, retriever, num_retrieve, excl_indices=None, return_indices=False):
        """Retrieve part-global features for st-classifier training
        Input:/home/dell/models/tarfoll_baselines/OnlineReID
            buffer
        Output:
            part_features: part-global features with shape of (part_nums, x, feature_dim)
            part_labels: labels with shape of (part_nums, x)
        """
        features = []
        labels = []
        indices = []
        for i in range(retriever.part_nums):
            part_indices = retriever.full_indexes[i].difference(buffer.buffer_tracker.remaining_indexes[i])
            part_indices = np.array(list(part_indices))
            num_retrieve = min(num_retrieve, part_indices.shape[0])
            part_indices = torch.from_numpy(np.random.choice(part_indices, num_retrieve, replace=False)).long()

            part_buffer_feature = getattr(buffer, "buffer_feature_{}".format(i))
            part_buffer_label = getattr(buffer, "buffer_label_{}".format(i))

            features.append(part_buffer_feature[part_indices])
            labels.append(part_buffer_label[part_indices])
            indices.append(part_indices)

        if return_indices:
            return features, labels, indices
        else:
            return features, labels

def class_balance_random_retrieve(buffer, num_retrieve, excl_indices=None, return_indices=False):
    if excl_indices is None:
        excl_indices = set()
    sample_ind = torch.tensor([], dtype=torch.long)
    # Use cache to retrieve indices belonging to each class in buffer
    for ind_set in buffer.buffer_tracker.class_index_cache.values():
        if ind_set:
            # Exclude some indices
            valid_ind = ind_set - excl_indices
            # Auxiliary indices for permutation
            perm_ind = torch.randperm(len(valid_ind))
            # Apply permutation, and select indices
            ind = torch.tensor(list(valid_ind), dtype=torch.long)[perm_ind][:(num_retrieve//2)]
            sample_ind = torch.cat((sample_ind, ind))

    # force pos-neg balance
    # if sample_ind.shape[0] != num_retrieve:
    #     sample_ind = torch.tensor([], dtype=torch.long)

    x = buffer.buffer_img[sample_ind]
    y = buffer.buffer_label[sample_ind]

    x = maybe_cuda(x)
    y = maybe_cuda(y)

    if return_indices:
        return x, y, sample_ind
    else:
        return x, y


        """Retrieve images and vis_map for lt-classifier training while keeping balance of pos and neg

        Output:
            imgs: for training with shape of (B, 3, H, W)
            labels: labels with shape of (B)
            vis_indicator: for visibility indication with shape of (B, 10)---front back 4 parts and 1 global
        """

        indices = []
        imgs = []
        labels = []
        vis_maps = []
        vis_indicators = []

        # [4, 9]
        for i in range(retriever.part_nums):
            # part_indices = self.full_indexes[i].difference(buffer.buffer_tracker.remaining_indexes[i])
            # if len(part_indices) == 0:
            #     continue
            part_indices = torch.tensor([], dtype=torch.long)
            for y_int in buffer.buffer_tracker.class_index_cache[i].keys():
                sample_part_indices = buffer.buffer_tracker.class_index_cache[i][y_int]
                if sample_part_indices:
                    perm_ind = torch.randperm(len(sample_part_indices))
                    sample_part_indices = torch.tensor(list(sample_part_indices), dtype=torch.long)[perm_ind][:(num_retrieve//2)]
                    # sample_part_indices = torch.tensor(list(sample_part_indices), dtype=torch.long)[:(num_retrieve//2)]
                    part_indices = torch.cat((part_indices, sample_part_indices))
            # force pos-neg balance
            if part_indices.shape[0] != num_retrieve:
                part_indices = torch.tensor([], dtype=torch.long)
            indices.append(part_indices)

            buffer_img = getattr(buffer, "buffer_img_{}".format(i))
            buffer_label = getattr(buffer, "buffer_label_{}".format(i))
            buffer_vis_indicator = getattr(buffer, "buffer_vis_indicator_{}".format(i))

            imgs.append(buffer_img[part_indices])
            labels.append(buffer_label[part_indices])
            vis_indicators.append(buffer_vis_indicator[part_indices])
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        vis_indicators = torch.concat(vis_indicators)

        if return_indices:
            return imgs, labels, vis_indicators, indices
        else:
            return imgs, labels, vis_indicators

def part_class_balance_random_retrieve(buffer, retriever, num_retrieve, excl_indices=None, return_indices=False):
        """Retrieve images and vis_map for lt-classifier training while keeping balance of pos and neg

        Output:
            imgs: for training with shape of (B, 3, H, W)
            labels: labels with shape of (B)
            vis_indicator: for visibility indication with shape of (B, 10)---front back 4 parts and 1 global
        """

        indices = []
        imgs = []
        labels = []
        vis_maps = []
        vis_indicators = []

        # [4, 9]
        for i in range(retriever.vis_map_nums, retriever.part_nums, retriever.part_nums//2):
            # part_indices = self.full_indexes[i].difference(buffer.buffer_tracker.remaining_indexes[i])
            # if len(part_indices) == 0:
            #     continue
            part_indices = torch.tensor([], dtype=torch.long)
            for y_int in buffer.buffer_tracker.class_index_cache[i].keys():
                sample_part_indices = buffer.buffer_tracker.class_index_cache[i][y_int]
                if sample_part_indices:
                    perm_ind = torch.randperm(len(sample_part_indices))
                    sample_part_indices = torch.tensor(list(sample_part_indices), dtype=torch.long)[perm_ind][:(num_retrieve//2)]
                    # sample_part_indices = torch.tensor(list(sample_part_indices), dtype=torch.long)[:(num_retrieve//2)]
                    part_indices = torch.cat((part_indices, sample_part_indices))
            # force pos-neg balance
            if part_indices.shape[0] != num_retrieve:
                part_indices = torch.tensor([], dtype=torch.long)
            indices.append(part_indices)

            buffer_img = getattr(buffer, "buffer_img_{}".format(i))
            buffer_label = getattr(buffer, "buffer_label_{}".format(i))
            buffer_vis_map = getattr(buffer, "buffer_vis_map_{}".format(i))
            buffer_vis_indicator = getattr(buffer, "buffer_vis_indicator_{}".format(i))

            imgs.append(buffer_img[part_indices])
            labels.append(buffer_label[part_indices])
            vis_maps.append(buffer_vis_map[part_indices])
            vis_indicators.append(buffer_vis_indicator[part_indices])
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        vis_maps = torch.concat(vis_maps)
        vis_indicators = torch.concat(vis_indicators)

        if return_indices:
            return imgs, labels, vis_maps, vis_indicators, indices
        else:
            return imgs, labels, vis_maps, vis_indicators

def class_balance_random_retrieve_features(buffer, num_retrieve, excl_indices=None, return_indices=False):
    if excl_indices is None:
        excl_indices = set()
    sample_ind = torch.tensor([], dtype=torch.long)
    # Use cache to retrieve indices belonging to each class in buffer
    for ind_set in buffer.buffer_tracker.class_index_cache.values():
        if ind_set:
            # Exclude some indices
            valid_ind = ind_set - excl_indices
            # Auxiliary indices for permutation
            perm_ind = torch.randperm(len(valid_ind))
            # Apply permutation, and select indices
            ind = torch.tensor(list(valid_ind), dtype=torch.long)[perm_ind][:(num_retrieve//2)]
            sample_ind = torch.cat((sample_ind, ind))

    # x = buffer.buffer_feature[sample_ind, :buffer.params.deep_feature_dim]  # only deep features
    if buffer.params.st_feature == "deep":
        x = buffer.buffer_feature[sample_ind, :buffer.params.deep_feature_dim]
    elif buffer.params.st_feature == "joint":
        x = buffer.buffer_feature[sample_ind, buffer.params.deep_feature_dim:]
    elif buffer.params.st_feature == "all":
        x = buffer.buffer_feature[sample_ind, :]
    else:
        return ValueError("Wrong feature")
    # x = buffer.buffer_feature[sample_ind, :]  # deep feature + joint features
    y = buffer.buffer_label[sample_ind]

    x = maybe_cuda(x)
    y = maybe_cuda(y)
    if return_indices:
        return x, y, sample_ind
    else:
        return x, y

def part_class_balance_random_features_retrieve(buffer, retriever, num_retrieve, excl_indices=None, return_indices=False):
        """Retrieve part-global features for st-classifier training while keeping balance of pos and neg
        Input:
            buffer
        Output:
            part_features: part-global features with shape of (part_nums, x, feature_dim)
            part_labels: labels with shape of (part_nums, x)
        """

        features = []
        labels = []
        indices = []
        for i in range(retriever.part_nums):
            part_indices = torch.tensor([], dtype=torch.long)
            for y_int in buffer.buffer_tracker.class_index_cache[i].keys():
                sample_part_indices = buffer.buffer_tracker.class_index_cache[i][y_int]
                if sample_part_indices:
                    perm_ind = torch.randperm(len(sample_part_indices))
                    sample_part_indices = torch.tensor(list(sample_part_indices), dtype=torch.long)[perm_ind][:(num_retrieve//2)]
                    part_indices = torch.cat((part_indices, sample_part_indices))

            part_buffer_feature = getattr(buffer, "buffer_feature_{}".format(i))
            part_buffer_label = getattr(buffer, "buffer_label_{}".format(i))

            features.append(part_buffer_feature[part_indices])
            labels.append(part_buffer_label[part_indices])
            indices.append(part_indices)

        if return_indices:
            return features, labels, indices
        else:
            return features, labels

def is_informative_st(buffer, current_tracklet:Tracklet, y, sliding_window_size, device="cpu"):
    """Judge current sample whether informative to be inserted by calculating the difference between current sample and local samples.
    """

    # Get features of current sample and buffer local samples including pre-defined and deep features

    # Judge whether to insert current feature based on local difference
    # 1) get local features; 2) get difference between current feature and local features; 3) judge whether to insert these feature to the lt-set
    y_int = y.item()
    sample_nums = buffer.buffer_tracker.class_num_cache[y_int]

    if sample_nums == 0:
        return True
    elif sample_nums <= sliding_window_size:
        sample_indexes = list(buffer.buffer_tracker.class_index_cache[y_int])
        # local_image_patches = buffer.buffer_img[sample_indexes]
        local_features = buffer.buffer_feature[sample_indexes]
    else:
        sample_indexes = list(buffer.buffer_tracker.class_index_cache[y_int])
        local_indexes = random.sample(sample_indexes, sliding_window_size)
        # current_index = buffer.buffer_tracker.current_index[y]
        # local_indexes = list(range(buffer.current_index-sliding_window_size, buffer.current_index))
        # def addition(n):
        #     if n < 0:
        #         n + buffer.buffer_img.size()
        #     return n
        # local_indexes = list(map(addition, local_indexes))
        local_features = buffer.buffer_feature[local_indexes]
        # local_image_patches = buffer.buffer_img[local_indexes]
    
    # current_df, local_df = deep_features(model, image_patch, image_patch.size(0), local_image_patches, local_image_patches.size(0))
    cosi = torch.nn.CosineSimilarity(dim=1)
    df_diff = torch.mean(1 - cosi(current_tracklet.deep_feature.unsqueeze(0), local_features[:, :128]))  # cosine distance
    # print("cosi: ", cosi(current_tracklet.deep_feature.unsqueeze(0), local_features[:, :128]))
    # bbox_diff = torch.abs(current_tracklet.bbox_feature - local_features[:, 128:130]).mean(0)  # L1 distance
    # joints_diff = distance.jensenshannon(current_tracklet.joints_feature.cpu().numpy(), local_features[:, 128:].cpu().numpy()).mean()  # distribution distance
    joints_diff = torch.mean(1 - cosi(current_tracklet.joints_feature, local_features[:, 128:]))  # cosine distance


    factor = buffer.memory_manager.factor[y_int]
    # diff = factor["appearance"]*df_diff + factor["bbox_height"]*bbox_diff[0,0] + factor["bbox_width"]*bbox_diff[0,1] + \
        #    factor["neck_height"]*joints_diff[0,0] + factor["waist_height"]*joints_diff[0,1] + factor["knee_height"]*joints_diff[0,2] + factor["ankle_height"]*joints_diff[0,3]

    diff = factor["appearance"]*df_diff + factor["pose"]*joints_diff
    # print("df_diff: {:.2f}*{:.3f}\t joints_diff: {:.2f}*{:.3f}\t diff: {:.3f}".format(factor["appearance"], df_diff, factor["pose"], joints_diff, diff))

    # diff = 0.5*df_diff + 0.5*joints_diff
    # print("df_diff: {:.2f}*{:.3f}\t joints_diff: {:.2f}*{:.3f}\t diff: {:.3f}".format(0.5, df_diff, 0.5, joints_diff, diff))

    if diff > factor["conf_thr"]:
        return True
    else:
        return False

def deep_features(model, current_x, n_current, local_x, n_local):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if n_current is None:
        num = n_current
        total_x = current_x
    else:
        num = n_current + n_local
        total_x = torch.cat((current_x, local_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    current_df = deep_features_[0:n_current]
    local_df = deep_features_[n_current:]
    return current_df, local_df

def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_

def add_minority_class_input(cur_x, cur_y, mem_size, num_class):
    """
    Find input instances from minority classes, and concatenate them to evaluation data/label tensors later.
    This facilitates the inclusion of minority class samples into memory when ASER's update method is used under online-class incremental setting.

    More details:

    Evaluation set may not contain any samples from minority classes (i.e., those classes with very few number of corresponding samples stored in the memory).
    This happens after task changes in online-class incremental setting.
    Minority class samples can then get very low or negative KNN-SV, making it difficult to store any of them in the memory.

    By identifying minority class samples in the current input batch, and concatenating them to the evaluation set,
        KNN-SV of the minority class samples can be artificially boosted (i.e., positive value with larger magnitude).
    This allows to quickly accomodate new class samples in the memory right after task changes.

    Threshold for being a minority class is a hyper-parameter related to the class proportion.
    In this implementation, it is randomly selected between 0 and 1 / number of all classes for each current input batch.


        Args:
            cur_x (tensor): current input data tensor.
            cur_y (tensor): current input label tensor.
            mem_size (int): memory size.
            num_class (int): number of classes in dataset.
        Returns
            minority_batch_x (tensor): subset of current input data from minority class.
            minority_batch_y (tensor): subset of current input label from minority class.
"""
    # Select input instances from minority classes that will be concatenated to pre-selected data
    threshold = torch.tensor(1).float().uniform_(0, 1 / num_class).item()

    # If number of buffered samples from certain class is lower than random threshold,
    #   that class is minority class
    cls_proportion = ClassBalancedRandomSampling.class_num_cache.float() / mem_size
    minority_ind = nonzero_indices(cls_proportion[cur_y] < threshold)

    minority_batch_x = cur_x[minority_ind]
    minority_batch_y = cur_y[minority_ind]
    return minority_batch_x, minority_batch_y


class BufferClassTracker(object):
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}

    def __init__(self, num_class, buffer_size=512, device="cpu"):
        super().__init__()
        # Initialize caches
        self.class_index_cache = defaultdict(set)
        self.class_num_cache = np.zeros(num_class)

        self.remaining_indexes = defaultdict(set(list(range(buffer_size))))


    def update_cache(self, buffer_y, new_y=None, ind=None, ):
        """
            Collect indices of buffered data from each class in set.
            Update class_index_cache with list of such sets.
                Args:
                    buffer_y (tensor): label buffer.
                    num_class (int): total number of unique class labels.
                    new_y (tensor): label tensor for replacing memory samples at ind in buffer.
                    ind (tensor): indices of memory samples to be updated.
                    device (str): device for tensor allocation.
        """

        # Get labels of memory samples to be replaced
        orig_y = buffer_y[ind]
        # Update caches
        for i, ny, oy in zip(ind, new_y, orig_y):
            oy_int = oy.item()
            ny_int = ny.item()
            # Update dictionary according to new class label of index i
            if oy_int in self.class_index_cache and i in self.class_index_cache[oy_int]:
                self.class_index_cache[oy_int].remove(i)
                self.class_num_cache[oy_int] -= 1
                self.remaining_indexes[oy_int].add(i)

            self.class_index_cache[ny_int].add(i)
            self.class_num_cache[ny_int] += 1
            # For discriminating st-term and lt-term
            # if i in self.remaining_indexes:
            self.remaining_indexes[ny_int].remove(i)


def get_future_step_parameters(model, params, grad_vector, grad_dims):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_model = copy.deepcopy(model)
    overwrite_grad(new_model.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_model.parameters():
            if param.grad is not None:
                param.data = param.data - params.learning_rate * param.grad.data
    return new_model

def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

class BufferPartClassTracker(object):
    # For faster label-based sampling (e.g., class balanced sampling), cache class-index via auxiliary dictionary
    # Store {class, set of memory sample indices from class} key-value pairs to speed up label-based sampling
    # e.g., {<cls_A>: {<ind_1>, <ind_2>}, <cls_B>: {}, <cls_C>: {<ind_3>}, ...}
    # part tracker
    # e.g., {<cls_A>: {<ind_1>: [<part_1>, <part_2>, -1, <part_4>, ...]}, <cls_B>: {<ind_2>: [<part_1>, <part_2>, -1, <part_4>]}, ...}

    def __init__(self, num_class, buffer_size=512, part_nums=10, vis_map_nums=4, device="cpu"):
        super().__init__()
        # Initialize caches
        self.part_nums = part_nums
        self.vis_map_nums = vis_map_nums
        self.class_index_cache = [{0: set(), 1: set()} for i in range(self.part_nums)] # [{0:inds, 1:inds} of head, {0:inds, 1:inds} of torso]
        self.part_index_cache = {i:{0: {}, 1: {}} for i in range(self.vis_map_nums, self.part_nums, self.part_nums//2)}  # [4, 9] or [2, 5]
        self.class_num_cache = np.zeros((num_class, self.part_nums))
        self.end_part_indexs = list(range(self.vis_map_nums, self.part_nums, self.part_nums//2))

        self.remaining_indexes = [set(list(range(buffer_size))) for _ in range(self.part_nums)]


    def update_cache(self, buffer, new_y=None, part_inds=None):
        """
            Collect indices of buffered data from each class in set.
            Update class_index_cache with list of such sets.
                Args:
                    buffer_y (tensor): label buffer.
                    num_class (int): total number of unique class labels.
                    new_y (tensor): label tensor for replacing memory samples at ind in buffer.
                    ind (tensor): indices of memory samples to be updated.
                    part_ind : indices of part indexes in the buffer to be updated. e.g., [1,12,13,-1,...] with shape of (part_nums)
                    device (str): device for tensor allocation.
        """
        for i, part_ind in enumerate(part_inds):
            if part_ind == -1:
                continue
            part_buffer_label = getattr(buffer, "buffer_label_{}".format(i))
            orig_y = part_buffer_label[part_ind]
            oy_int = orig_y.item()
            ny_int = new_y.item()

            # remove operation
            if oy_int in self.class_index_cache[i] and part_ind in self.class_index_cache[i][oy_int]:
                self.class_index_cache[i][oy_int].remove(part_ind)
                self.class_num_cache[oy_int, i] -= 1
                self.remaining_indexes[i].add(part_ind)
            
            # update operation
            self.class_index_cache[i][ny_int].add(part_ind)
            self.class_num_cache[ny_int, i] += 1
            if part_ind in self.remaining_indexes[i]:
                self.remaining_indexes[i].remove(part_ind)
            
            # Add parts relationship
            if i in self.part_index_cache.keys():
                if oy_int in self.part_index_cache[i] and part_ind in self.part_index_cache[i][oy_int]:
                    self.part_index_cache[i][oy_int].pop(part_ind)
                self.part_index_cache[i][ny_int][part_ind] = part_inds
    
    def remove_cache(self, oy_int, vis_index, idxs_removed):
        # Get labels of memory samples to be replaced
        for idx_removed in idxs_removed:
            part_inds = self.part_index_cache[vis_index][oy_int][idx_removed]
            for i, part_ind in enumerate(part_inds):
                if part_ind == -1:
                    continue
                # remove operation
                if oy_int in self.class_index_cache[i] and part_ind in self.class_index_cache[i][oy_int]:
                    self.class_index_cache[i][oy_int].remove(part_ind)
                    self.class_num_cache[oy_int, i] -= 1
                    self.remaining_indexes[i].add(part_ind)

            # remove part_inds
            if oy_int in self.part_index_cache[vis_index] and idx_removed in self.part_index_cache[vis_index][oy_int]:
                self.part_index_cache[vis_index][oy_int].pop(idx_removed)
    
    def print_class_nums(self):
        print("front pos/neg:{}/{}".format(self.class_num_cache[1][self.end_part_indexs[0]], self.class_num_cache[0][self.end_part_indexs[0]]))
        if len(self.end_part_indexs) > 1:
            print("back pos/neg:{}/{}".format(self.class_num_cache[1][self.end_part_indexs[1]], self.class_num_cache[0][self.end_part_indexs[1]]))

    
    def return_class_nums(self):
        if len(self.end_part_indexs) > 1:
            return self.class_num_cache[1][self.end_part_indexs[0]], self.class_num_cache[0][self.end_part_indexs[0]], self.class_num_cache[1][self.end_part_indexs[1]], self.class_num_cache[0][self.end_part_indexs[1]]
        return self.class_num_cache[1][self.end_part_indexs[0]], self.class_num_cache[0][self.end_part_indexs[0]]
    
    def return_part_nums(self, part_idx):
        return self.class_num_cache[0][part_idx], self.class_num_cache[1][part_idx]
            

    def check_tracker(self):
        print(self.class_num_cache.sum())
        print(len([k for i in self.class_index_cache.values() for k in i]))
    
    def sample(self, buffer_x, buffer_y, n_smp_cls, excl_indices=None, device="cpu"):
        """
            Take same number of random samples from each class from buffer.
                Args:
                    buffer_x (tensor): data buffer.
                    buffer_y (tensor): label buffer.
                    n_smp_cls (int): number of samples to take from each class.
                    excl_indices (set): indices of buffered instances to be excluded from sampling.
                    device (str): device for tensor allocation.
                Returns
                    x (tensor): class balanced random sample data tensor.
                    y (tensor): class balanced random sample label tensor.
                    sample_ind (tensor): class balanced random sample index tensor.
        """
        if excl_indices is None:
            excl_indices = set()

        # Get indices for class balanced random samples
        # cls_ind_cache = class_index_tensor_list_cache(buffer_y, num_class, excl_indices, device=device)

        sample_ind = torch.tensor([], device=device, dtype=torch.long)

        # Use cache to retrieve indices belonging to each class in buffer
        for ind_set in self.class_index_cache.values():
            if ind_set:
                # Exclude some indices
                valid_ind = ind_set - excl_indices
                # Auxiliary indices for permutation
                perm_ind = torch.randperm(len(valid_ind), device=device)
                # Apply permutation, and select indices
                ind = torch.tensor(list(valid_ind), device=device, dtype=torch.long)[perm_ind][:n_smp_cls]
                sample_ind = torch.cat((sample_ind, ind))

        x = buffer_x[sample_ind]
        y = buffer_y[sample_ind]

        x = maybe_cuda(x)
        y = maybe_cuda(y)

        return x, y, sample_ind