import tensorflow as tf
import numpy as np
import sys
import time

tf.compat.v1.disable_eager_execution()

def print_output(final_rank):

	rank_idx = np.argsort(final_rank[:,0], axis=0)

	# reverse the array to get descending order
	rank_idx_asc = rank_idx[::-1]
	print("1. Printing top 20 node ids with their ranks")
	print("S No. \t Node Id \t Rank")
	for i in range(20):
		print(i+1, "\t" , rank_idx_asc[i], "\t" , final_rank[rank_idx_asc[i]][0])
    
def pagerank(adj_matrix, num_nodes, teleport_prob=0.15, min_err=1.0e-3):
    beta = 1 - teleport_prob
    e = np.ones((num_nodes, 1)) * teleport_prob / num_nodes

    v = tf.compat.v1.placeholder(tf.float32, shape=[num_nodes, 1])
    M = adj_matrix / tf.sparse.reduce_sum(adj_matrix, axis=0)

    pagerank = tf.add(tf.sparse.sparse_dense_matmul(M, v) * beta, e)
    diff_in_rank = tf.reduce_sum(tf.abs(pagerank - v))

    init = tf.compat.v1.global_variables_initializer()
    traversed_edges_counter = 0
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        R = np.ones((num_nodes, 1)) / num_nodes
        while True:
            new_pagerank = sess.run(pagerank, feed_dict={v: R})
            err_norm = sess.run(diff_in_rank, feed_dict={pagerank: new_pagerank, v: R})
            R = new_pagerank
            traversed_edges_counter += adj_matrix.indices.shape[0]
            if err_norm < min_err:
                break
    return R,traversed_edges_counter

def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            data.append((node1, node2))
    
    max_nodeid = max(max(data, key=lambda x: x[0])[0], max(data, key=lambda x: x[1])[1]) + 1
    adj_matrix = tf.sparse.SparseTensor(indices=[[node2, node1] for node1, node2 in data], values=[1.0]*len(data), dense_shape=[max_nodeid, max_nodeid])
    
    return adj_matrix, max_nodeid

if __name__ == '__main__':
    filepath = sys.argv[1]
    
    start_time = time.time()    
    adj_matrix, num_nodes = load_data(filepath)

    calculation_time = time.time()
    pagerank_vector,traversed_edges = pagerank(adj_matrix, num_nodes)
    end_time = time.time()

    execution_time = end_time - calculation_time
    all_time = end_time - start_time

    GTEPS = traversed_edges / execution_time / 1e9
    print_output(pagerank_vector)
    print("Pagerank calculation took", all_time, "seconds")
    print("GTEPS:", GTEPS)