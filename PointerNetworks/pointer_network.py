#!/usr/bin/env python

import math
import numpy as np
import random
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib import  rnn
# See https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264

# Uncomment this to stop corner printing and see full/verbatim
#np.set_printoptions(threshold=np.nan)

attn_size=10
batch_size=1024
max_q_len=0
bias=True
def generate_nested_sequence(length, min_seglen=5, max_seglen=10):
    """Generate low-high-low sequence, with indexes of the first/last high/middle elements"""

    # Low (1-5) vs. High (6-10)
    seq_before = [(random.randint(1,5)) for x in range(random.randint(min_seglen, max_seglen))]
    seq_during = [(random.randint(6,10)) for x in range(random.randint(min_seglen, max_seglen))]
    seq_after = [random.randint(1,5) for x in range(random.randint(min_seglen, max_seglen))]
    seq = seq_before + seq_during + seq_after
    seq_len=len(seq)
    # Pad it up to max len with 0's
    seq = seq + ([0] * (length - len(seq)))
    return [seq, len(seq_before), len(seq_before) + len(seq_during)-1,len(seq),seq_len]


def create_one_hot(length, index):
    """Returns 1 at the index positions; can be scaled by client"""
    a = np.zeros([length])
    a[index] = 1.0
    return a


def get_lstm_state(cell):
    """Centralize definition of 'state', to swap .c and .h if desired"""
    return cell.c


def print_pointer(arr, first, second):
    """Pretty print the array, along with pointers to the first/second indices"""
    first_string = " ".join([(" " * (2 - len(str(x))) + str(x)) for x in arr])
    print(first_string)
    second_array = ["  "] * len(arr)
    second_array[first] = "^1"
    second_array[second] = "^2"
    if (first == second):
        second_array[first] = "^b"
    second_string = " " + " ".join([x for x in second_array])
    print(second_string)


def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None):
    with tf.variable_scope(scope, reuse = reuse): #1st input=(batch size,num of question,2*atten_size)
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] == batch_size:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if bias:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1]) #score=(batch size, num of words)
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now

def mask_attn_score(score, memory_sequence_length, score_mask_value=-1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)



# def pointer_net(question, question_len, cell, params, scope = "pointer_network"):
def pointer_net(encoder_outputs,initial_state, weights_p,cell,memorry,scope="pointer_network"):
    '''
    Answer pointer network as proposed in https://arxiv.org/pdf/1506.03134.pdf.

    Args:
        passage:        RNN passage output from the bidirectional readout layer (batch_size, timestep, dim)
        passage_len:    variable lengths for passage length
        question:       RNN question output of shape (batch_size, timestep, dim) for question pooling
        question_len:   Variable lengths for question length
        cell:           rnn cell of type RNN_Cell.
        params:         Appropriate weight matrices for attention pooling computation

    Returns:
        softmax logits for the answer pointer of the beginning and the end of the answer span
    '''
    with tf.variable_scope(scope):
        # weights_q, weights_p = params
        inputs = [encoder_outputs, initial_state.h]
        p1_logits = attention(inputs, attn_size, weights_p ,memory_len=memorry, scope = "attention")
        scores = tf.expand_dims(p1_logits, -1)
        attention_pool = tf.reduce_sum(scores,1)
        _, state = cell(attention_pool, initial_state)
        inputs = [encoder_outputs, state.h]
        p2_logits = attention(inputs, attn_size, weights_p, memory_len=memorry,scope = "attention", reuse = True)
        return tf.stack((p1_logits,p2_logits),1)

def outputs_fun(points_logits):
    logit_1, logit_2 = tf.split(points_logits, 2, axis = 1)
    logit_1 = tf.transpose(logit_1, [0, 2, 1])
    dp = tf.matmul(logit_1, logit_2)
    dp = tf.matrix_band_part(dp, 0, 15)
    output_index_1 = tf.argmax(tf.reduce_max(dp, axis = 2), -1)
    output_index_2 = tf.argmax(tf.reduce_max(dp, axis = 1), -1)
    output_index = tf.stack([output_index_1,output_index_2], axis = 1)
    return output_index
    # self.output_index = tf.argmax(self.points_logits, axis = 2)
def cross_entropy(output, target):
    cross_entropy = target * tf.log(output + 1e-8)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2) # sum across passage timestep
    cross_entropy = tf.reduce_mean(cross_entropy, 1) # average across pointer networks output
    return tf.reduce_mean(cross_entropy) # average across batch size
#
# def loss_function(input,indices,points_logits):
#     with tf.variable_scope("loss"):
#         shapes = input.shape
#         indices_prob = tf.one_hot(indices, shapes[1])
#         mean_loss = cross_entropy(points_logits,indices_prob)
#     return mean_loss



def evaluate(max_length,         # J
             batch_size,         # B
             lstm_width,         # L
             num_blend_units,    # D
             num_training_loops,
             loss_interval,
             optimizer):
    """Core evaluation function given hyperparameters -- returns tuple of training losses and test percentage"""

    # S: Size of each vector (1 here, ignored/implicit)
    # I: num_indices (2 here; start and end)
    # J: input length (40 max here = max_length *), following notation of Vinyals (2015)
    # B: batch_size (param *)
    # L: lstm_width* units
    # D: Blend units


    num_indices = 2                         # I
                       # S  (dimensions per token)
    input_length = max_length               # J again
    generation_value = 12

    training_segment_lengths = (11, 20)     # Each of the low/high/low segment lengths
    testing_segment_lengths = (6, 10)       # "", but with no overlap whatsoever with the training seg lens

    reset_params = {"steps": 3000, "loss": .03}
    attention_size=lstm_width
    # Initialization parameters
    m = 0.0
    s = 0.5
    init = tf.random_normal_initializer(m, s)



    # Cleanup on aisle 6
    tf.reset_default_graph()

    # Training data placeholders
    inputs = tf.placeholder(tf.int32, name="ptr-in", shape=(batch_size, input_length))      # B x J
    seq_len = tf.placeholder(tf.int32, name="ptr-in", shape=(batch_size,))
    # The one hot (over J) distributions, by batch and by index (start=1 and end=2)
    actual_index_dists = tf.placeholder(tf.float32,                                           # I x B x J
                                        name="ptr-out",
                                        shape=(num_indices, batch_size, input_length))
    actual_index_dists_=tf.transpose(actual_index_dists,[1,0,2])
    # Define the type of recurrent cell to be used. Only used for sizing.
    cell_enc = tf.contrib.rnn.LSTMCell(lstm_width,
                                       use_peepholes=False,
                                       initializer=init)




    # ###################  ENCODER

    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[13, lstm_width-2])
    with tf.variable_scope("rnn_encoder"):

        # 1-layer LSTM with n_hidden units.
        rnn_cell = rnn.LSTMCell(lstm_width,use_peepholes=False,
                                       initializer=init)
        # inputs=tf.reshape(inputs,[],dtype)
        inputs_embedding=tf.nn.embedding_lookup(embeddings,inputs)
        # inputs_g=tf.expand_dims(inputs,-1)
        # generate prediction
        outputs, enc_states = tf.nn.dynamic_rnn(rnn_cell, inputs=inputs_embedding, sequence_length=seq_len,dtype=tf.float32)

    # Need a dummy output to point on it. End of decoding.
    encoder_outputs =   outputs
    cell_dec = rnn.LSTMCell(lstm_width,
                                       use_peepholes=False,
                                       initializer=init)

    # ###################  DECODER
    # special symbol is max_length, which can never come from the actual data
    starting_generation_symbol = tf.constant(generation_value,                              # B x S
                                             shape=(batch_size,
                                                    ),
                                             dtype=tf.int32)
    #
    # input_dimensions=1
    # W_d_in = tf.get_variable("W_d_in", [input_dimensions, lstm_width], initializer=init)  # S x L
    # b_d_in = tf.get_variable("b_d_in", [batch_size, lstm_width], initializer=init)  # B x L
    # cell_input = tf.nn.elu(tf.matmul(starting_generation_symbol, W_d_in) + b_d_in)  # B x L
    #
    # output, dec_state = cell_dec(cell_input, enc_states)




    weights_e=tf.get_variable(name="weights_e", dtype=tf.float32,
                                 shape=[lstm_width, attention_size],initializer=init)
    weights_d=tf.get_variable(name="weights_d", dtype=tf.float32,
                                 shape=[lstm_width, attention_size],initializer=init)
    weights_v=tf.get_variable(name="weights_v", dtype=tf.float32,
                                 shape=[attention_size,],initializer=init)
    weights_p=([weights_e,weights_d],weights_v)


    with tf.variable_scope("rnn_decoder"):
        point_logit=pointer_net(encoder_outputs,enc_states, weights_p,cell_dec,seq_len,scope="pointer_network")

        idx_distributions=tf.transpose(point_logit,[1,0,2])
        output_index=outputs_fun(point_logit)
    # ############## LOSS
    # RMS of difference across all batches, all indices
    with tf.variable_scope("loss"):
        loss = cross_entropy(point_logit,actual_index_dists_)
        # loss=tf.sqrt(tf.reduce_mean(tf.pow(idx_distributions - actual_index_dists, 2.0)))
    train = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()  # config=config)
    sess.run(init_op)

    # ############## TRAINING
    train_dict = {}
    sequences = []
    first_indexes = []
    second_indexes = []
    sen_len=[]
    train_data=[]
    # Note that our training/testing datasets are the same size as our batch. This is
    #   unusual and just makes the code slightly simpler. In general your dataset size
    #   is >> your batch size and you rotate batches from the dataset through.
    for batch_index in range(batch_size):
        data = generate_nested_sequence(max_length,
                                        training_segment_lengths[0],
                                        training_segment_lengths[1])
        train_data.append(data)
        sequences.append(data[0])                                           # J
        first_indexes.append(create_one_hot(input_length, data[1]))         # J
        second_indexes.append(create_one_hot(input_length, data[2]))        # J
        sen_len.append(data[4])
    train_dict[inputs] = np.stack(sequences)                                # B x J
    train_dict[actual_index_dists] = np.stack([np.stack(first_indexes),     # I x B x J
                                      np.stack(second_indexes)])
    train_dict[seq_len]=sen_len

    losses = []
    for step in range(num_training_loops):
        tf_outputs = [loss, train, idx_distributions, actual_index_dists]
        results = sess.run(tf_outputs, feed_dict=train_dict)
        step_loss = results[0]

        if step % loss_interval == 0:
            losses.append(step_loss)
            print("%s: %s" % (step, step_loss))
            sys.stdout.flush()
        # if step >= reset_params["steps"] and step_loss > reset_params["loss"]:
        #     return None

    # ############## TESTING
    print(" === TEST === ")

    sequences = []
    first_indexes = []
    second_indexes = []
    sen_len = []
    for batch_index in range(batch_size):
        data = generate_nested_sequence(max_length,
                                        testing_segment_lengths[0],
                                        testing_segment_lengths[1])
        sequences.append(data[0])                                           # J
        first_indexes.append(create_one_hot(input_length, data[1]))         # J
        second_indexes.append(create_one_hot(input_length, data[2]))        # J
        sen_len.append(data[4])
    test_dict = {inputs: np.stack(sequences),
                 actual_index_dists: np.stack([np.stack(first_indexes),
                                               np.stack(second_indexes)]),
                 seq_len:sen_len}
    # 0 is loss, 1 is prob dists, 2 is actual one-hots
    results = sess.run([loss, idx_distributions, actual_index_dists,output_index], feed_dict=test_dict)

    print('test loss {}'.format(results[0]))
    incorrect_pointers = 0
    for batch_index in range(batch_size):

        first_diff = first_indexes[batch_index] - results[1][0][batch_index]
        first_diff_max = np.max(np.abs(first_diff))
        first_ptr = np.argmax(results[1][0][batch_index])

        if first_diff_max >= .5:  # bit stricter than argmax but let's hold ourselves to high standards, people
            incorrect_pointers += 1
        second_diff = second_indexes[batch_index] - results[1][1][batch_index]
        second_diff_max = np.max(np.abs(second_diff))
        second_ptr = np.argmax(results[1][1][batch_index])
        if second_diff_max >= .5:
            incorrect_pointers += 1
            # print(np.max(results[1][0][batch_index]), np.max(results[1][1][batch_index]), first_ptr, second_ptr,results[2][batch_index])
        print_pointer(sequences[batch_index], first_ptr, second_ptr)
        print("")

    test_pct = np.round(100.0 * ((2 * batch_size) - incorrect_pointers) / (2 * batch_size), 5)
    print("")
    print(" %s / %s (correct/total); test pct %s" % ((2*batch_size) - incorrect_pointers,
                                                     2 * batch_size,
                                                     test_pct))

    #==============

    # ====================
    print(" === train === ")
    sequences = []
    first_indexes = []
    second_indexes = []
    sen_len = []
    for batch_index, data in enumerate(train_data):
        sequences.append(data[0])  # J
        first_indexes.append(create_one_hot(input_length, data[1]))  # J
        second_indexes.append(create_one_hot(input_length, data[2]))  # J
        sen_len.append(data[4])
    test_dict = {inputs: np.stack(sequences),
                 actual_index_dists: np.stack([np.stack(first_indexes),
                                               np.stack(second_indexes)]),
                 seq_len: sen_len}
    # 0 is loss, 1 is prob dists, 2 is actual one-hots
    results = sess.run([loss, idx_distributions, actual_index_dists], feed_dict=test_dict)

    incorrect_pointers = 0
    for batch_index in range(batch_size):

        first_diff = first_indexes[batch_index] - results[1][0][batch_index]
        first_diff_max = np.max(np.abs(first_diff))
        first_ptr = np.argmax(results[1][0][batch_index])
        if first_diff_max >= .5:  # bit stricter than argmax but let's hold ourselves to high standards, people
            incorrect_pointers += 1
        second_diff = second_indexes[batch_index] - results[1][1][batch_index]
        second_diff_max = np.max(np.abs(second_diff))
        second_ptr = np.argmax(results[1][1][batch_index])
        if second_diff_max >= .5:
            incorrect_pointers += 1

        print_pointer(sequences[batch_index], first_ptr, second_ptr)
        print("")

    train_pct = np.round(100.0 * ((2 * batch_size) - incorrect_pointers) / (2 * batch_size), 5)
    print("")
    print(" %s / %s (correct/total); train pct %s" % ((2 * batch_size) - incorrect_pointers,
                                                     2 * batch_size,
                                                     test_pct))
    sys.stdout.flush()
    return losses, test_pct,train_pct


max_reset_retries = 20
for reset_loop_index in range(max_reset_retries):

    # Create optimizer - AdaGrad works well on this problem
    learning_rate =0.2
    adagrad_optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    lstm_blend = 10
    result = evaluate(max_length=60,
                      batch_size=1024,
                      lstm_width=lstm_blend,
                      num_blend_units=lstm_blend,
                      num_training_loops=5000,
                      loss_interval=100,
                      optimizer=adagrad_optimizer)

    if result is None:
        print("Warning: loss is stagnant-- starting again")
    else:
        print("Training losses: %s" % (str(result[0])))
        print("Test percentage: %s" % (result[1]))
        print("train percentage: %s" % (result[2]))
        break  # We're done!
