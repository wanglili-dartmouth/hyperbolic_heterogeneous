from __future__ import print_function

import os
import h5py
import multiprocessing 
import re
import argparse
import json
import sys
import random
import numpy as np
import networkx as nx
import pandas as pd
import glob
import gc
import seaborn as sns
from scipy.sparse import identity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import average_precision_score, roc_auc_score
from keras.layers import Input, Layer, Dense, Embedding
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, TerminateOnNaN, TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping
import functools
import fcntl
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer

from hyperboloid.utils import hyperboloid_to_poincare_ball, load_data, load_embedding
from hyperboloid.utils import perform_walks, determine_positive_and_negative_samples
from hyperboloid.losses import  hyperbolic_softmax_loss, hyperbolic_sigmoid_loss, hyperbolic_hypersphere_loss
from hyperboloid.generators import TrainingDataGenerator
from hyperboloid.visualise import draw_graph, plot_degree_dist
from hyperboloid.callbacks import Checkpointer
from tqdm import tqdm
import random
import math
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)
        
def lock_method(lock_filename):
    ''' Use an OS lock such that a method can only be called once at a time. '''

    def decorator(func):

        @functools.wraps(func)
        def lock_and_run_method(*args, **kwargs):

            # Hold program if it is already running 
            # Snippet based on
            # http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
            fp = open(lock_filename, 'r+')
            done = False
            while not done:
                try:
                    fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    done = True
                except IOError:
                    pass
            return func(*args, **kwargs)

        return lock_and_run_method

    return decorator         
def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
    lock_method(lock_filename)(fn)(*args, **kwargs)        
def save_test_results(filename, seed, data, ):
    d = pd.DataFrame(index=[seed], data=data)
    if os.path.exists(filename):
        test_df = pd.read_csv(filename, sep=",", index_col=0)
        test_df = d.combine_first(test_df)
    else:
        test_df = d
    test_df.to_csv(filename, sep=",")
def threadsafe_save_test_results(lock_filename, filename, seed, data):
    threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)
K.set_floatx("float64")
K.set_epsilon(1e-15)
np.set_printoptions(suppress=True)

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

config.log_device_placement=False
config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
def minkowki_dot(u, v):
    """
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:,:rank].dot(v[:,:rank].T)
    return euc_dp - u[:,rank, None] * v[:,rank]
def hyperbolic_distance_hyperboloid(u, v):
    mink_dp = minkowki_dot(u, v)
    mink_dp = np.maximum(-1 - mink_dp, 1e-15)
    return np.arccosh(1 + mink_dp)

def gans_to_hyperboloid(x):
    t = K.sqrt(1. + K.sum(K.square(x), axis=-1, keepdims=True))
    return tf.concat([x, t], axis=-1)

def euclidean_dot(x, y):
    axes = len(x.shape) - 1, len(y.shape) - 1
    return K.batch_dot(x, y, axes=axes)

def minkowski_dot(x, y):
    axes = len(x.shape) - 1, len(y.shape) -1
    return K.batch_dot(x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)

def hyperboloid_initializer(shape, r_max=1e-3):

    def poincare_ball_to_hyperboloid(X, append_t=True):
        x = 2 * X
        t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
        if append_t:
            x = K.concatenate([x, t], axis=-1)
        return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

    def sphere_uniform_sample(shape, r_max):
        num_samples, dim = shape
        X = tf.random_normal(shape=shape, dtype=K.floatx())
        X_norm = K.sqrt(K.sum(K.square(X), axis=-1, keepdims=True))
        U = tf.random_uniform(shape=(num_samples, 1), dtype=K.floatx())
        return r_max * U ** (1./dim) * X / X_norm

    w = sphere_uniform_sample(shape, r_max=r_max)
    # w = tf.random_uniform(shape=shape, minval=-r_max, maxval=r_max, dtype=K.floatx())
    return poincare_ball_to_hyperboloid(w)

class HyperboloidEmbeddingLayer(Layer):
    
    def __init__(self, 
        num_nodes, 
        embedding_dim, 
        **kwargs):
        super(HyperboloidEmbeddingLayer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.embedding = self.add_weight(name='embedding', 
          shape=(self.num_nodes, self.embedding_dim),
          initializer=hyperboloid_initializer,
          trainable=True)
        super(HyperboloidEmbeddingLayer, self).build(input_shape)

    def call(self, idx):

        embedding = tf.gather(self.embedding, idx)
        # embedding = tf.nn.embedding_lookup(self.embedding, idx)

        return embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim + 1)
    
    def get_config(self):
        base_config = super(HyperboloidEmbeddingLayer, self).get_config()
        base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})
        return base_config

class ExponentialMappingOptimizer(optimizer.Optimizer):
    
    def __init__(self, 
        lr=0.1, 
        use_locking=False,
        name="ExponentialMappingOptimizer"):
        super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
        self.lr = lr

    def _apply_dense(self, grad, var):
        spacial_grad = grad[:,:-1]
        t_grad = -1 * grad[:,-1:]
        
        ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1)
        tangent_grad = self.project_onto_tangent_space(var, ambient_grad)
        
        exp_map = self.exponential_mapping(var, - self.lr * tangent_grad)
        
        return tf.assign(var, exp_map)
        
    def _apply_sparse(self, grad, var):
        indices = grad.indices
        values = grad.values

        p = tf.gather(var, indices, name="gather_apply_sparse")
        # p = tf.nn.embedding_lookup(var, indices)

        spacial_grad = values[:, :-1]
        t_grad = -values[:, -1:]

        ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1, name="optimizer_concat")

        tangent_grad = self.project_onto_tangent_space(p, ambient_grad)
        exp_map = self.exponential_mapping(p, - self.lr * tangent_grad)

        return tf.scatter_update(ref=var, indices=indices, updates=exp_map, name="scatter_update")
    
    def project_onto_tangent_space(self, hyperboloid_point, minkowski_ambient):
        return minkowski_ambient + minkowski_dot(hyperboloid_point, minkowski_ambient) * hyperboloid_point
   
    def exponential_mapping( self, p, x ):

        def normalise_to_hyperboloid(x):
            return x / K.sqrt( -minkowski_dot(x, x) )

        norm_x = K.sqrt( K.maximum(np.float64(0.), minkowski_dot(x, x) ) ) 
        ####################################################
        exp_map_p = tf.cosh(norm_x) * p
        
        idx = tf.cast( tf.where(norm_x > K.cast(0., K.floatx()), )[:,0], tf.int64)
        non_zero_norm = tf.gather(norm_x, idx)
        z = tf.gather(x, idx) / non_zero_norm

        updates = tf.sinh(non_zero_norm) * z
        dense_shape = tf.cast( tf.shape(p), tf.int64)
        exp_map_x = tf.scatter_nd(indices=idx[:,None], updates=updates, shape=dense_shape)
        
        exp_map = exp_map_p + exp_map_x 
        #####################################################
        # z = x / K.maximum(norm_x, K.epsilon()) # unit norm 
        # exp_map = tf.cosh(norm_x) * p + tf.sinh(norm_x) * z
        #####################################################
        exp_map = normalise_to_hyperboloid(exp_map) # account for floating point imprecision

        return exp_map

def build_model(num_nodes, args):

    x = Input(shape=(1 + 1 + args.num_negative_samples, ), 
        name="model_input", 
        dtype=tf.int32)
    y = HyperboloidEmbeddingLayer(num_nodes, args.embedding_dim, name="embedding_layer")(x)
    model = Model(x, y)

    return model


def load_weights(model, args):

    previous_models = sorted(glob.iglob(
        os.path.join(args.embedding_path, "*.csv.gz")))
    if len(previous_models) > 0:
        model_file = previous_models[-1]
        initial_epoch = int(model_file.split("/")[-1].split("_")[0])
        print ("previous models found in directory -- loading from file {} and resuming from epoch {}".format(model_file, initial_epoch))
        embedding_df = load_embedding(model_file)
        embedding = embedding_df.reindex(sorted(embedding_df.index)).values
        model.layers[1].set_weights([embedding])
    else:
        print ("no previous model found in {}".format(args.embedding_path))
        initial_epoch = 0

    return model, initial_epoch

def parse_args():
    '''
    parse args from the command line
    '''
    parser = argparse.ArgumentParser(description="hyperboloid algorithm for feature learning on complex networks")

    parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
        help="edgelist to load.")
    parser.add_argument("--features", dest="features", type=str, default=None,
        help="features to load.")
    parser.add_argument("--labels", dest="labels", type=str, default=None,
        help="path to labels")

    parser.add_argument("--seed", dest="seed", type=int, default=0,
        help="Random seed (default is 0).")
    parser.add_argument("--lr", dest="lr", type=np.float64, default=1.,
        help="Learning rate (default is 1.).")

    parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=5,
        help="The number of epochs to train for (default is 5).")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
        help="Batch size for training (default is 50).")
    parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=20, 
        help="Number of negative samples for training (default is 10).")
    parser.add_argument("--context-size", dest="context_size", type=int, default=5,
        help="Context size for generating positive samples (default is 3).")
    parser.add_argument("--patience", dest="patience", type=int, default=10,
        help="The number of epochs of no improvement in loss before training is stopped. (Default is 10)")

    parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
        help="Dimension of embeddings for each layer (default is 2).", default=2)

    parser.add_argument("-p", dest="p", type=float, default=1.,
        help="node2vec return parameter (default is 1.).")
    parser.add_argument("-q", dest="q", type=float, default=1.,
        help="node2vec in-out parameter (default is 1.).")
    parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
        help="Number of walks per source (default is 10).")
    parser.add_argument('--walk-length', dest="walk_length", type=int, default=80, 
        help="Length of random walk from source (default is 80).")

    parser.add_argument("--sigma", dest="sigma", type=np.float64, default=1.,
        help="Width of gaussian (default is 1).")

    parser.add_argument("--alpha", dest="alpha", type=float, default=0, 
        help="Probability of randomly jumping to a similar node when walking.")

    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
        help="Use this flag to set verbosity of training.")
    parser.add_argument('--workers', dest="workers", type=int, default=2, 
        help="Number of worker threads to generate training patterns (default is 2).")

    parser.add_argument("--walks", dest="walk_path", default=None, 
        help="path to save random walks.")

    parser.add_argument("--embedding", dest="embedding_path", default=None, 
        help="path to save embedings.")

    parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

    parser.add_argument('--use-generator', action="store_true", help='flag to train using a generator')

    parser.add_argument('--visualise', action="store_true", 
        help='flag to visualise embedding (embedding_dim must be 2)')

    parser.add_argument('--no-walks', action="store_true", 
        help='flag to only train on edgelist (no random walks)')

    parser.add_argument('--all-negs', action="store_true", 
        help='flag to only train using all nodes as negative samples')

    parser.add_argument("--time", dest="time_threshold", type=float, default=1, 
        help="Probability of randomly walk to past")
    parser.add_argument("--test-results-dir", dest="test_results_dir",  
        help="path to save results.")
    args = parser.parse_args()
    return args

def configure_paths(args):
    '''
    build directories on local system for output of model after each epoch
    '''


    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
        print ("making {}".format(args.embedding_path))
    print ("saving embedding to {}".format(args.embedding_path))

def get_het_dic(DATA):
    if DATA=='DBLP':
        heterg_dictionary={'a': ['p'], 'p': ['a', 'c'], 'c': ['p']}
    elif DATA=='Movie':
        heterg_dictionary={'d': ['m'], 'm': ['a', 'd'], 'a': ['m']}
    else:
        print('Datasets error')
    return heterg_dictionary
def dblp_generation(G, path_length, heterg_dictionary, m, start):#生成一条just walks
    path = []   
    path.append(start)
    no_next_types = 0    
    heterg_probability = 0
    heterg_dictionary=get_het_dic('Movie')
    count = {}
    for i in heterg_dictionary:
        count[i]=0

    while len(path) < path_length:

        cur = path[-1]#获得上一个节点 node_type,node_name       
        count[cur[0]]+=1
        next_type_options=[e for e in G[cur]]
        if not next_type_options:
            break
        mini_count={}
        for i in heterg_dictionary:
            mini_count[i]=0
        for e in next_type_options:
            mini_count[e[0]]+=1
        prob=[]
        sum_=0.0
        for i in count.keys():
            sum_+=math.exp(-count[i])
        for e in next_type_options:
            prob.append(math.exp(-count[e[0]])/mini_count[e[0]]/sum_ )
        prob=np.array(prob)
        prob /=prob.sum()
        #print(cur)
        #print(next_type_options)
        #print(mini_count)
        #print(prob)
        #print("-----------------")
        next_node = np.random.choice(next_type_options,p=prob)
        path.append(next_node)
    return path
def generate_walks(G, num_walks, walk_length, m, heterg_dictionary):
    print('Generating walks .. ')
    walks = []
    nodes = list(G.nodes())
    for cnt in tqdm(range(num_walks)):
        random.shuffle(nodes)
        for node in nodes:
            just_walks = dblp_generation(G, walk_length, heterg_dictionary, m, start=node)
            #print(just_walks)
            walks.append(just_walks)
    print('Walks done .. ')
    return walks
def evaluate_rank_and_AP(scores, 
    edgelist, non_edgelist):
    assert not isinstance(edgelist, dict)
    assert (scores <= 0).all()

    if not isinstance(edgelist, np.ndarray):
        edgelist = np.array(edgelist)

    if not isinstance(non_edgelist, np.ndarray):
        non_edgelist = np.array(non_edgelist)

    edge_scores = scores[edgelist[:,0], edgelist[:,1]]
    non_edge_scores = scores[non_edgelist[:,0], non_edgelist[:,1]]

    labels = np.append(np.ones_like(edge_scores), 
        np.zeros_like(non_edge_scores))
    scores_ = np.append(edge_scores, non_edge_scores)
    ap_score = average_precision_score(labels, scores_) # macro by default
    auc_score = roc_auc_score(labels, scores_)
        
    idx = (-non_edge_scores).argsort()
    ranks = np.searchsorted(-non_edge_scores, 
        -edge_scores, sorter=idx) + 1
    ranks = ranks.mean()

    print ("MEAN RANK =", ranks, "AP =", ap_score, 
        "AUROC =", auc_score)

    return ranks, ap_score, auc_score
def main():
    args = parse_args()
    print ("Configured paths")
    graph = nx.read_weighted_edgelist(args.edgelist)
    heterg_dictionary=get_het_dic('Movie')
    np.random.seed(0)
    walks =  generate_walks(graph, 10, 80, 0, heterg_dictionary)
    graph=nx.convert_node_labels_to_integers(graph,first_label=0,label_attribute='Name')
    Name={}
    for d in graph.nodes:
        Name[graph.nodes[d]['Name']]=d
    print(walks[0])
    walks = [ [Name[w] for w in w_list] for w_list in walks]
    print(walks[0])
    nodes_list_p=[]
    nodes_list_c=[]
    nodes_list_a=[]
    for d in graph.nodes:
        if graph.nodes[d]['Name'][0]=='a':
            nodes_list_a.append(d)
        if graph.nodes[d]['Name'][0]=='m' :
            nodes_list_p.append(d)    
        if graph.nodes[d]['Name'][0]=='d' :
            nodes_list_c.append(d)   
    
    print(len(nodes_list_a))
    print(len(nodes_list_c))
    print(len(nodes_list_p))
    test_edges = np.array(list(graph.subgraph(nodes_list_p+nodes_list_a).edges()))
    #print(test_edges)
    test_non_edges = []
    for p in nodes_list_p:
        for a in nodes_list_a:
            if not graph.has_edge(p,a):
                test_non_edges.append([p,a])
    test_non_edges=np.array(test_non_edges)
    print(len(test_edges))
    print(len(test_non_edges))
    print("--------------------------------")

    #print(walks)
    
    

    assert not (args.visualise and args.embedding_dim > 2), "Can only visualise two dimensions"
    assert args.embedding_path is not None, "you must specify a path to save embedding"


    if not os.path.exists(args.embedding_path):

        configure_paths(args)

        
        # build model
        num_nodes = len(graph)
        
        model = build_model(num_nodes, args)
        model, initial_epoch = load_weights(model, args)
        optimizer = ExponentialMappingOptimizer(lr=args.lr)
        loss = hyperbolic_softmax_loss(sigma=args.sigma)
        model.compile(optimizer=optimizer, 
            loss=loss, 
            target_tensors=[tf.placeholder(dtype=tf.int32)])
        model.summary()

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="loss", 
                patience=args.patience, 
                verbose=True),
            Checkpointer(epoch=initial_epoch, 
                nodes=sorted(graph.nodes()), 
                embedding_directory=args.embedding_path)
        ]            

        positive_samples, negative_samples, probs = \
                determine_positive_and_negative_samples(graph,walks,args)

      
        if args.use_generator:
            print ("Training with data generator with {} worker threads".format(args.workers))
            training_generator = TrainingDataGenerator(positive_samples,  
                    probs,
                    model,
                    args)

            model.fit_generator(training_generator, 
                workers=args.workers,
                max_queue_size=10, 
                use_multiprocessing=args.workers>0, 
                epochs=args.num_epochs, 
                steps_per_epoch=len(training_generator),
                initial_epoch=initial_epoch, 
                verbose=args.verbose,
                callbacks=callbacks
            )

        else:
            print ("Training without data generator")

            train_x = np.append(positive_samples, negative_samples, axis=-1)
            train_y = np.zeros([len(train_x), 1, 1], dtype=np.int32 )

            model.fit(train_x, train_y,
                shuffle=True,
                batch_size=args.batch_size, 
                epochs=args.num_epochs, 
                initial_epoch=initial_epoch, 
                verbose=args.verbose,
                callbacks=callbacks
            )

        print ("Training complete")
        embedding = model.get_weights()[0]
    else:
        sep = ","
        header = "infer"
        embedding_df = pd.read_csv(args.embedding_path+"/00005_embedding.csv.gz",
        sep=sep, header=header, index_col=0)
        embedding_df = embedding_df.reindex(sorted(embedding_df.index))    
        # row 0 is embedding for node 0
        # row 1 is embedding for node 1 etc...
        embedding = embedding_df.values
        print ("Loading complete")

    
#    embedding = hyperboloid_to_poincare_ball(embedding)
    dists = hyperbolic_distance_hyperboloid(embedding, embedding)
    scores = -dists
    test_results = dict()
    (mean_rank_recon, ap_recon, roc_recon) = evaluate_rank_and_AP(scores, test_edges, test_non_edges)

    test_results.update({"mean_rank_recon": mean_rank_recon, 
        "ap_recon": ap_recon,
        "roc_recon": roc_recon})
    test_results_dir = args.test_results_dir
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    test_results_filename = os.path.join(test_results_dir, "test_results.csv")
    test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")
    touch(test_results_lock_filename)

    print ("saving test results to {}".format(test_results_filename))

    threadsafe_save_test_results(test_results_lock_filename, test_results_filename, args.seed, data=test_results )

    print ("done")
if __name__ == "__main__":
    main()