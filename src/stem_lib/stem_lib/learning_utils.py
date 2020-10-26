import time
import queue
from collections import deque
import threading
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .stdlib.collections import dotdict
from .stdlib.bimapper import BiMapper
from .models import lstm
from .layers import duplicate


def triplet_loss(alpha=0.2):
    # 先に訓練データ(過去のデータ)とそれに対応する教師(正負)のセットを作る
    # そうすると, できるはず
    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : [ positive, negative ]
        """
        # print(y_pred, y_true)

        # Note:
        #   Error:
        #       # " slice index 1 of dimension 0 out of bounds"
        #           >>>
        #             For me, the issue resolved when I changed for example, y_pred[0] to y_pred[:,0,:]. You might need to change your implementation suitably.
        #           >>>
        #           https://gist.github.com/JustinhoCHN/a243056d87f4c2728de9e7ea4923de01
        #
        anchor = y_pred[:, 0, :]
        positive = y_true[:, 0, :]
        negative = y_true[:, 1, :]

        pos_dist = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(anchor, positive)), 1)
        neg_dist = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(anchor, negative)), 1)

        basic_loss = tf.math.add(tf.math.subtract(pos_dist, neg_dist), alpha)
        loss = tf.math.reduce_mean(tf.math.maximum(basic_loss, 0.0), 0)
        return loss

    return loss


def select_triplets(anchor_embedding, positive_embeddings, negative_embeddings, alpha=0.2):
    triplets = []
    neg_dists = tf.math.reduce_sum(tf.math.square(
        anchor_embedding - negative_embeddings), 1)
    pos_size = tf.shape(positive_embeddings)[0]
    for pos_idx in range(pos_size):
        pos_dist = tf.math.reduce_sum(tf.math.square(
            anchor_embedding - positive_embeddings[pos_idx]))
        neg_indices = np.where(neg_dists - pos_dist < alpha)[0]
        nrof_neg_indices = tf.shape(neg_indices)[0]
        if nrof_neg_indices > 0:
            rnd_idx = np.random.randint(nrof_neg_indices)
            neg_idx = neg_indices[rnd_idx]
            triplets.append(
                [anchor_embedding, positive_embeddings[pos_idx], negative_embeddings[neg_idx]])

    np.random.shuffle(triplets)
    return triplets

def make_model(sensor_data_queue_size, sensor_data_segment_size):
    inputs = tf.keras.Input(
        shape=(sensor_data_queue_size, sensor_data_segment_size),
        batch_size=None
    )
    base_model = lstm.Model()
    x = base_model(inputs)
    outputs = duplicate.Layer()(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss=triplet_loss(), optimizer='adam')
    
    base_model.summary()
    model.summary()

    return model


def recal_embeddings(model, replay_buffer):
    frames = []
    indices = []
    
    for index, frame, embedding in replay_buffer.popleft_each():
        indices.append(index)
        frames.append(frame)
    
    embeddings, _ = model(np.array(frames))

    for index, frame, embedding in zip(indices, frames, embeddings):
        replay_buffer.append(index, frame, embedding)

def append_initial_frames(frames, replay_buffer, model, state_classifier):
    embeddings, _ = model(np.array(frames))
    clustered_ids = state_classifier.fit_predict(embeddings)
    for frame, embedding, clustered_id in zip(frames, embeddings, clustered_ids):
        replay_buffer.append(clustered_id, frame, embedding)

def estimate_state(frame, model, state_classifier):
    [embedding], _ = model(np.array([frame]))
    [state_id] = state_classifier.predict([embedding])

    return state_id, frame, embedding

class StateNameIdBiMapper():
    def __init__(self):
        # x: name; y: id
        self.bimapper = BiMapper()
    
    def get_name(self, state_id):
        if state_id not in self.bimapper.y2x:
            return None

        return self.bimapper.y2x[state_id]
    

    def get_id(self, state_name):
        if state_name not in self.bimapper.x2y:
            return None
        
        return self.bimapper.x2y[state_name]
        
    def bind(self, state_name, state_id):
        self.bimapper.bind(state_name, state_id)


class ReplayBuffer():
    def __init__(self, n_states, queue_maxlen):
        self.n_states = n_states
        
        self.feature_frames = [deque(maxlen=queue_maxlen) for _ in range(n_states)]
        self.feature_frames_back = [deque(maxlen=queue_maxlen) for _ in range(n_states)]
        self.embeddings = [deque(maxlen=queue_maxlen) for _ in range(n_states)]
        self.embeddings_back = [deque(maxlen=queue_maxlen) for _ in range(n_states)]
    
    def length(self):
        return sum([len(frame) for frame in self.feature_frames])

    def append(self, state_major_id, feature_frame, embedding):
        self.feature_frames[state_major_id].append(feature_frame)
        self.embeddings[state_major_id].append(embedding)
    
    def append_back_buffer(self, state_major_id, feature_frame, embedding):
        self.feature_frames_back[state_major_id].append(feature_frame)
        self.embeddings_back[state_major_id].append(embedding)

    def get_feature_frames(self, condition=None):
        return self._get_samples(self.feature_frames, condition)
    
    def get_embeddings(self, condition=None):
        return self._get_samples(self.embeddings, condition)
    
    def iterate(self):
        for index in range(self.n_states):
            for frame, embedding in zip(self.feature_frames[index], self.embeddings[index]):
                yield index, frame, embedding
    
    def popleft_each(self):
        for index in range(self.n_states):
            while self.feature_frames[index] and self.embeddings[index]:
                frame = self.feature_frames[index].popleft()
                embedding = self.embeddings[index].popleft()
                yield index, frame, embedding

    def swap_buffer(self):
        self.feature_frames, self.feature_frames_back = self.feature_frames_back, self.feature_frames
        self.embeddings, self.embeddings_back = self.embeddings_back, self.embeddings
        
        self.feature_frames_back.clear()
        self.embeddings_back.clear()

    def _get_samples(self, source, condition=None):
        """
        condition : function
            if you want to get positves
            ```
                lambda idx: idx == anchor_index
            ```
            or negatives
            ```
                lambda idx: idx != anchor_index
            ```
        """
        samples = []
        indices = []
        for idx, queue in enumerate(source):
            for ele in queue:
                if condition is None or condition(idx):
                    samples.append(ele)
                    indices.append(idx)
        return samples, indices
    

# class StateClassifier():
#     def __init__(self, n_states):
#         self.n_states = n_states
#         self.kmeans = KMeans(
#             n_clusters=n_states
#         )

#     def update_cluster_centers(self, embeddings):
#         if len(embeddings) < self.n_states:
#             return None
        
#         clustered_ids = kmeans.fit_predict(embeddings)

    
#     def get_nearest_cluster_index(self, embedding):
#         l = [[np.linalg.norm(self.cluster_centers[i] - embedding), i] for i in range(len(self.cluster_centers))]

#         l.sort(key=lambda x: x[0])

#         return l[1]

#     @property
#     def cluster_centers(self):
#         return self.kmeans.cluster_centers_
        
# class Estimator():
#     def __init__(self, model=None, precedents_dict=None):
#         self.is_running = False
#         self.results = queue.Queue()
#         self.model = model
#         self.inputs = None
#         self.precedents_dict = precedents_dict
#         self._prev_centers = None

#     def run(self, inputs=None, supervised_state_label=None):
#         if self.is_running:
#             return
#         self.is_running = True
#         self.inputs = inputs
#         self.supervised_state_label = supervised_state_label
#         self.worker_thread = threading.Thread(target=self.worker)
#         self.worker_thread.start()

#     def worker(self):
#         [embedding], _ = self.model(self.inputs)

#         result = dotdict({
#             'inputs': None,
#             'embedding': None,
#             'estimated_state': None,
#             'clustered': None,
#             'cluster_centers': None
#         })
#         n_states = len(self.precedents_dict) - 1
#         result.inputs = self.inputs
#         result.embedding = embedding
#         result.estimated_state = n_states  # not categorized state
#         result.supervised_state_label = self.supervised_state_label

#         embeddings = []
#         for precedents in self.precedents_dict:
#             embeddings.extend(
#                 [precedent.embedding for precedent in precedents])
#         embeddings.append(embedding)

#         if len(embeddings) > n_states:
#             kmeans = KMeans(
#                 n_clusters=n_states
#             )

#             result.clustered = kmeans.fit_predict(embeddings)

#             id_remap = [i for i in range(len(kmeans.cluster_centers_))]

#             if self._prev_centers is not None:
#                 result.cluster_centers = [None] * len(kmeans.cluster_centers_)
#                 pairs = nearest(kmeans.cluster_centers_, self._prev_centers)

#                 for ifrom, ito in pairs:
#                     id_remap[ifrom] = ito
#                     result.cluster_centers[ito] = kmeans.cluster_centers_[
#                         ifrom]
#             else:
#                 result.cluster_centers = kmeans.cluster_centers_

#             self._prev_centers = result.cluster_centers
#             result.estimated_state = id_remap[result.clustered[-1]]
#             result.id_remap = id_remap

#             # print(result.clustered[-1])
#             # print(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])

#         self.results.put(result)
#         self.is_running = False


class Trainor():
    def __init__(self, precedents_dict=None):
        self.is_running = False
        self.precedents_dict = precedents_dict
        self.results = queue.Queue()

    def run(self, model=None, anchor=None):
        if self.is_running:
            return
        self.is_running = True
        self.model = model
        self.anchor = anchor
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()

    def worker(self):
        positive_embeddings = []
        negative_embeddings = []
        anchor_state = get_major_state(self.anchor)

        for state, precedents in enumerate(self.precedents_dict):
            if state == len(self.precedents_dict) - 1:
                break
            if state == anchor_state:
                positive_embeddings.extend(
                    [precedent.embedding for precedent in precedents])
            else:
                negative_embeddings.extend(
                    [precedent.embedding for precedent in precedents])

        if len(positive_embeddings) <= 0 or len(negative_embeddings) <= 0:
            self.is_running = False
            return

        triplets = select_triplets(
            self.anchor.embedding, positive_embeddings, negative_embeddings)
        if len(triplets) <= 0:
            self.is_running = False
            return
        inputs = [self.anchor.inputs[0]] * len(triplets)
        targets = []
        for anchor, positive, negative in triplets:
            targets.append([positive, negative])

        inputs = np.array(inputs)
        targets = np.array(targets)

        self.model.fit(inputs, targets, batch_size=1)
        self.is_running = False


def nearest(a, b):
    na, nb = len(a), len(b)

    # Combinations of a and b
    comb = product(range(na), range(nb))

    # [[distance, index number(a), index number(b)], ... ]
    l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]

    # Sort with distance
    l.sort(key=lambda x: x[0])

    pairs = []
    for _ in range(min(na, nb)):
        m, ia, ib = l[0]
        pairs.append([ia, ib])
        # Remove items with same index number
        l = list(filter(lambda x: x[1] != ia and x[2] != ib, l))

    return pairs


def get_major_state(segment):
    state = segment.get('supervised_state')
    if state is None:
        state = segment.get('estimated_state')

    return state


def make_visualized_graph_plots(precedents_dict, current_segment):
    def make_plot(segment, reduced, min_pos, max_pos):
        plot = dotdict()
        plot.position = np.array(reduced)
        if min_pos[0] is None or min_pos[0] > reduced[0]:
            min_pos[0] = reduced[0]
        if min_pos[1] is None or min_pos[1] > reduced[1]:
            min_pos[1] = reduced[1]
        if max_pos[0] is None or max_pos[0] < reduced[0]:
            max_pos[0] = reduced[0]
        if max_pos[1] is None or max_pos[1] < reduced[1]:
            max_pos[1] = reduced[1]
        
        plot.estimated_state = segment.get('estimated_state')
        plot.supervised_state = segment.get('supervised_state')

        return plot, min_pos, max_pos

    plots = []
    meta = dotdict()
    embeddings = []

    for precedents in precedents_dict:
        embeddings.extend([precedent.embedding for precedent in precedents])

    if len(embeddings) < 2:
        return None, None

    if current_segment is not None:
        embeddings.append(current_segment.embedding)
        
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    meta.min, meta.max = [None, None], [None, None]

    idx = 0
    for precedents in precedents_dict:
        for precedent in precedents:
            plot, meta.min, meta.max = make_plot(precedent, reduced[idx], meta.min, meta.max)
            plots.append(plot)
            idx += 1

    if current_segment is not None:
        plot, meta.min, meta.max = make_plot(current_segment, reduced[idx], meta.min, meta.max)
        plots.append(plot)

    meta.min = np.array(meta.min)
    meta.max = np.array(meta.max)
    return meta, plots
