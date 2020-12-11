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
def dblp_generation(G, path_length, heterg_dictionary, m, start):
    path = []   
    path.append(start)
    no_next_types = 0    
    heterg_probability = 0
    heterg_dictionary=get_het_dic('DBLP')
    count = {}
    for i in heterg_dictionary:
        count[i]=0

    while len(path) < path_length:

        cur = path[-1]
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
def plot(embeddings):
    X=[]
    Y=[]
    Label=[]
    for i in range(len(embeddings)):
        X.append(embeddings[i][0])
        Y.append(embeddings[i][1])
    canvas_height = 15
    canvas_width = 15
    dot_size = 10
    text_size = 18
    legend_setting = False #“brief” / “full” / False


    sns.set(style="whitegrid")

    # set canvas height & width
    plt.figure(figsize=(canvas_width, canvas_height))

    color_paltette=[(223,186,36)]
    pts_colors=list(range(len(embeddings)))
    for i in range(len(embeddings)):
        pts_colors[i]="color_1"

    for i in range(len(color_paltette)):
        color_paltette[i] = (color_paltette[i][0] / 255, color_paltette[i][1] / 255, color_paltette[i][2] / 255)
        
        
    # reorganize dataset
    draw_dataset = {'x': X,
                    'y': Y, 
                    'label':list(range(len(embeddings))),
                    'ptsize': dot_size,
                    "cpaltette": color_paltette,
                    'colors':pts_colors}

    #draw scatterplot points
    ax = sns.scatterplot(x = "x",y = "y", alpha = 1,s = draw_dataset["ptsize"],hue="colors", palette=draw_dataset["cpaltette"], legend = legend_setting, data = draw_dataset)


    return ax
def hyperbolic_distance_poincare(u, v):
	assert len(u.shape) == len(v.shape)
	norm_u = np.linalg.norm(u, keepdims=False, axis=-1)
	norm_u = np.minimum(norm_u, np.nextafter(1,0, ))
	norm_v = np.linalg.norm(v, keepdims=False, axis=-1)
	norm_v = np.minimum(norm_v, np.nextafter(1,0, ))
	uu = np.linalg.norm(u - v, keepdims=False, axis=-1, ) ** 2
	dd = (1 - norm_u**2) * (1 - norm_v**2)
	return np.arccosh(1 + 2 * uu / dd)
def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + np.sum(np.square(X), axis=-1, keepdims=True)
	x = np.concatenate([x, t], axis=-1)
	return 1 / (1. - np.sum(np.square(X), axis=-1, keepdims=True)) * x
def main():
    args = parse_args()
    print ("Configured paths")
    graph = nx.read_weighted_edgelist(args.edgelist)
    heterg_dictionary=get_het_dic('DBLP')
    np.random.seed(0)
    graph=nx.convert_node_labels_to_integers(graph,first_label=0,label_attribute='Name')



    sep = ","
    header = "infer"
    embedding_df = pd.read_csv(args.embedding_path+"/00005_embedding.csv.gz",
    sep=sep, header=header, index_col=0)
    embedding_df = embedding_df.reindex(sorted(embedding_df.index))    
    # row 0 is embedding for node 0
    # row 1 is embedding for node 1 etc...
    embedding = embedding_df.values
    print ("Loading complete")
    embedding = hyperboloid_to_poincare_ball(embedding)
    zero=np.array([0,0])
    sum2=0
    tot2=0
    sum4=0
    tot4=0
    sum6=0
    tot6=0
    new_embedding=[]
    for i in range(len(embedding)):
        if(graph.nodes[i]['Name'][0]=='a'):
            if(hyperbolic_distance_poincare(embedding[i],zero)<2):
                sum2+=graph.degree(i)
                new_embedding.append(embedding[i])
                tot2+=1
            elif(hyperbolic_distance_poincare(embedding[i],zero)>4):
                sum6+=graph.degree(i)
                new_embedding.append(embedding[i])
                tot6+=1
            else:
                sum4+=graph.degree(i)
                new_embedding.append(embedding[i])
                tot4+=1
    ax=plot(new_embedding)
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color="black", linewidth=2)
    d2 = math.sqrt(1/(2/(math.cosh(2)-1)+1))
    d4 = math.sqrt(1/(2/(math.cosh(4)-1)+1))
    d6 = math.sqrt(1/(2/(math.cosh(6)-1)+1))

    print((sum2+0.0)/tot2)
    print((sum4+0.0)/tot4)
    print((sum6+0.0)/tot6)
    print(tot2)
    print(tot4)
    print(tot6)
    print(len(embedding))
    d2x=x*d2
    d2y=y*d2
    d4x=x*d4
    d4y=y*d4
    d6x=x*d6
    d6y=y*d6

    for i in range(len(embedding)):

        if(graph.nodes[i]['Name']=='a7277'): #Wei Wang
            ax.plot(embedding[i][0], embedding[i][1], 'ro')
            ax.text(embedding[i][0], embedding[i][1]+0.05, "Wei Wang" ,horizontalalignment='center',verticalalignment='center', size=15, color='black', weight='semibold')
        if(graph.nodes[i]['Name']=='a19922'): #H. V. Jagadish
            ax.plot(embedding[i][0], embedding[i][1], 'ro')
            ax.text(embedding[i][0], embedding[i][1]-0.05, "H. V. Jagadish" ,horizontalalignment='center',verticalalignment='center', size=15, color='black', weight='semibold')
        if(graph.nodes[i]['Name']=='a113162'): #Surajit Chaudhuri
            ax.plot(embedding[i][0], embedding[i][1], 'ro')
            ax.text(embedding[i][0], embedding[i][1]-0.05, "Surajit Chaudhuri" ,horizontalalignment='center',verticalalignment='center', size=15, color='black', weight='semibold')
        if(graph.nodes[i]['Name']=='a113755'): #Christos Faloutsos
            ax.plot(embedding[i][0], embedding[i][1], 'ro')
            ax.text(embedding[i][0], embedding[i][1]-0.05, "Christos Faloutsos" ,horizontalalignment='center',verticalalignment='center', size=15, color='black', weight='semibold')
            
        if(graph.nodes[i]['Name']=='a16696'): #Philip S. Yu
            ax.plot(embedding[i][0], embedding[i][1], 'ro')
            ax.text(embedding[i][0], embedding[i][1]+0.05, "Philip S. Yu" ,horizontalalignment='center',verticalalignment='center', size=15, color='black', weight='semibold')
            
    ax.plot(d2x, d2y, color="purple", linewidth=1.5)
    ax.plot(d4x, d4y, color="purple", linewidth=1.5)
    ax.plot(d6x, d6y, color="purple", linewidth=1.5)
    ax.axis("equal")
    ax.figure.savefig("embedding.pdf",bbox_inches='tight')
    print ("done")
if __name__ == "__main__":
    main()


