from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
flags = tf.app.flags

FLAGS = flags.FLAGS
config =''
word_embedding=''
def assign_pretrained_word_embedding(params):
    print("using pre-trained word emebedding.started.word2vec_model_path:",FLAGS.fast_text)
    import fastText as ft
    word2vec_model = ft.load_model(FLAGS.fast_text)

    vocab_size=params["vocab_size"]
    word_embedding_final = np.zeros((vocab_size,FLAGS.embedding_size))  # create an empty word_embedding list.

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}

    for (word,idx) in vocab.items():
        embedding=word2vec_model.get_word_vector(word)
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_final[idx,:] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_final[idx, :] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.

    # word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor

    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    return word_embedding_final

def ini():
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    global config
    with open(json_path) as f:
        config = json.load(f)
    params = {
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }
    global word_embedding
    word_embedding = assign_pretrained_word_embedding(params)

def parse_exmp2(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([FLAGS.sentence_max_len], tf.int64),
            "_id": tf.FixedLenFeature([], tf.int64),
            # "text": tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    labels = feats.pop('label')

    return feats, labels

def pred_input_fn(filenames, shuffle_buffer_size,shuffle=True,repeat=0):
    # dataset = tf.data.TFRecordDataset(filenames) filename is a string
    print('tfrecord')
    print(filenames)
    files = tf.data.Dataset.list_files(filenames, shuffle=shuffle)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    dataset = dataset.map(parse_exmp2, num_parallel_calls=10)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if repeat>0:
        dataset = dataset.repeat(repeat).batch(FLAGS.batch_size)
    else:
        dataset = dataset.repeat().batch(FLAGS.batch_size)

    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset

def pred_per_file(file_path,my_model):

    input_fn_for_pred = lambda: pred_input_fn(file_path, shuffle_buffer_size=0,shuffle=False,repeat=1)
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    with open(json_path) as f:
        config = json.load(f)
    params = {
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }

    params['word_embedding']=word_embedding
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=tf.estimator.RunConfig(keep_checkpoint_max=2,model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    eval_spec = classifier.predict(input_fn=input_fn_for_pred)

    count = 0

    result=[]
    label=[]
    prob=[]
    for e in eval_spec:
        count += 1
        result.append((file_path,count,int(list(e.get('classes', ''))[0])))
        label.append(int(list(e.get('classes', ''))[0]))
        prob.append(e.get('probabilities'))
    # for r in result:
    #     print(r)

    return label,prob

def real_one(input_filename):
    label_list=[]
    id_list=[]
    for serialized_example in tf.python_io.tf_record_iterator(input_filename):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        label = example.features.feature["label"]
        features = example.features.feature["_id"]
        label_list.append(label.int64_list.value[0])
        id_list.append(features.int64_list.value[0])
    return label_list,id_list


def pred(my_model,fg,out_name):
    ini()
    global FLAGS
    FLAGS=fg
    import re
    import numpy as np
    import pandas as pd
    from collections import Counter
    acc_list=[]
    from collections import defaultdict
    dic = defaultdict(list)
    local_path, pattern = os.path.split(FLAGS.pred_file_pattern)
    for root, dirs, files in os.walk(local_path):
        for file in files:

            regu_cont = re.compile(r'.{}'.format(pattern), re.I)
            if regu_cont.match(file):
                right = 0
                wrong = 0
                print(file)
                file_path = os.path.join(root, file)
                # file_path='/home/tom/new_data/input_data/train_shuf_100000_11000_eval.tfrecords'
                label,prob=pred_per_file(file_path,my_model)
                label_list,id_list=real_one(file_path)
                for pl,_lb,_id in zip(prob,label_list,id_list):
                    dic[_id].append(pl)

    prob_np=np.zeros((max(dic.keys())+1,19))
    assert prob_np.shape[0]==102276+1
    for (k, v_list) in dic.items():
        v_arry=np.array(v_list)
        prob_np[k,:]=np.mean(v_arry,0)

    max_id=np.argmax(prob_np,1)+1
    _idx=[i for i in range((max(dic.keys())+1))]
    mx_id_list=list(max_id)
    np.save('../../output/{}.np'.format(out_name),prob_np)
    pd_dic={'id':_idx,
     'class':mx_id_list
     }

    pd.DataFrame.from_dict(pd_dic)[["id","class"]].to_csv('../../output/sub_{}.csv'.format(out_name),index=None)

def pred_eval(unused_argv):
    import re
    acc_list=[]
    local_path, pattern = os.path.split(FLAGS.pred_file_pattern)
    for root, dirs, files in os.walk(local_path):
        for file in files:

            regu_cont = re.compile(r'.{}'.format(pattern), re.I)
            if regu_cont.match(file):
                right = 0
                wrong = 0
                print(file)
                file_path = os.path.join(root, file)
                # file_path='/home/tom/new_data/input_data/train_shuf_100000_11000_eval.tfrecords'
                label,acc=pred_per_file(file_path)
                label_list,id_list=real_one(file_path)
                from collections import defaultdict
                dic=defaultdict(list)
                for pl,rl,_id in zip(label,label_list,id_list):
                    dic[_id].append(pl==rl)
                    if pl==rl:
                        right+=1
                    else:
                        wrong+=1
                # print(dic)
                print(right,wrong,acc)
                right = 0
                wrong = 0
                for v_list in dic.values():
                    tn = 0
                    fn = 0
                    for v in v_list:
                        if v:
                            tn += 1
                        else:
                            fn += 1
                        if tn > fn:
                            right += 1
                        elif tn < fn:
                            wrong += 1
                        elif tn == fn:

                            if v_list[0]:
                                right += 1
                            else :
                                wrong += 1
                print(right, wrong, right/(right+wrong))
                print('gg')
                acc_list.append((acc,right/(right+wrong)))
    for a in acc_list:
        print(a)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=pred(my_model,fg,out_name))

