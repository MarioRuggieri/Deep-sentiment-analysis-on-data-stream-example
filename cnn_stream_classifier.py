from __future__ import print_function
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import itertools
import re
from optparse import OptionParser
from bigdl.dataset import news20
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
import time

def get_train_data(train_file):
    texts = []
    nlines = 0
    with open(train_file) as f:
        for line in f:
            text_label = re.split(r'\t+', line)
            text_label[0] = int(text_label[0])+1
            texts.append((text_label[1],text_label[0]))
            nlines = nlines+1
    return (texts,nlines)

def get_test_data(test_file):
    texts = []
    nlines = 0
    with open(test_file) as f:
        for line in f:
            texts.append((line,1))
    return texts

def text_to_words(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words

# calculate the frequency of words in each text corpus, 
# sort by frequency (max to min)
# and assign an id to each word
def analyze_texts(data_rdd):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()  #return a list

# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l

def to_vec(token, b_w2v, embedding_dim):
    if token in b_w2v:
        return b_w2v[token]
    else:
        return pad([], 0, embedding_dim)

def to_sample(vectors, label, embedding_dim):
    # flatten nested list
    flatten_features = list(itertools.chain(*vectors))
    # a row for each word vector
    features = np.array(flatten_features, dtype='float').reshape([sequence_len, embedding_dim])
    features = features.transpose(1, 0)
    return Sample.from_ndarray(features, np.array(label))

def build_cnn(class_num, input_dim, hidden_dim):
    #each row is an input vector for the RNN
    model = Sequential()
    model.add(Reshape([input_dim, 1, sequence_len]))
    model.add(SpatialConvolution(input_dim, hidden_dim, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(SpatialConvolution(hidden_dim, hidden_dim, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(Reshape([hidden_dim]))
    model.add(Linear(hidden_dim, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model

def preprocess_texts(data_rdd, sequence_len, max_words, embedding_dim, streaming):
    #get list of (word, (index,freq)) representing the training set
    word_to_ic = analyze_texts(data_rdd)

    #at most max_words words among the most frequent
    if streaming:
        word_to_ic = dict(word_to_ic)
    else:
        word_to_ic = dict(word_to_ic[10: max_words])
    
    bword_to_ic = sc.broadcast(word_to_ic)

    #prepare and broadcast word embeddings filtered through word_to_ic 
    w2v = news20.get_glove_w2v(dim=embedding_dim)
    filtered_w2v = {w: v for w, v in w2v.items() if w in word_to_ic}
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    #get a list of words for each line + label in data_rdd
    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0]) if
                                w in bword_to_ic.value], text_label[1]))

    #pad lists of words to sequence_len size + label
    padded_tokens_rdd = tokens_rdd.map( lambda tokens_label: 
                                        (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))

    #get vectors from words + label
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens_label[0]], tokens_label[1]))

    #get matrix sample composed by word vectors for each text
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))

    return sample_rdd

def train(sc,
          train_data,
          class_num,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split, checkpoint_path, logdir, app_name):
    print('Processing text dataset')

    #generate RDD and sample RDD for the optimizer
    train_data_rdd = sc.parallelize(train_data, 2)
    sample_rdd = preprocess_texts(train_data_rdd, sequence_len, max_words, embedding_dim, False)

    #split into training and validation set
    train_rdd, val_rdd = sample_rdd.randomSplit(
        [training_split, 1-training_split])

    optimizer = Optimizer(
        model=build_cnn(class_num, input_dim=embedding_dim, hidden_dim=128),
        training_rdd=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method=Adagrad(learningrate=0.01, learningrate_decay=0.0002))

    optimizer.set_validation(
        batch_size=batch_size,
        val_rdd=val_rdd,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )

    #set checkpoint path for model
    optimizer.set_checkpoint(EveryEpoch(), checkpoint_path)
    #set train and val log for tensorboard
    train_summary = TrainSummary(log_dir=logdir, app_name=app_name)
    val_summary = ValidationSummary(log_dir=logdir, app_name=app_name)
    optimizer.set_train_summary(train_summary)
    optimizer.set_val_summary(val_summary)

    return optimizer.optimize()

def map_predict_label(l):
    return np.array(l).argmax()

def classify_stream(rdd_test, train_model):
    if not(rdd_test.isEmpty()):
        #probability vectors, one for each input
        predictions = train_model.predict(rdd_test).collect()
        #get max probability indices
        y_pred = np.array([ map_predict_label(s) for s in predictions])
        for y in y_pred:
            if y==0: 
                print('NEGATIVE\n')
            else:
                print('POSITIVE\n')

def show(rdd_test):
    for elem in rdd_test.collect():
        print(elem[0].encode('utf-8'),end='')

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")  # noqa
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="15")
    parser.add_option("-c", "--checkpoint_path", dest="checkpoint_path", default="checkpoint")
    parser.add_option("-l", "--log_path", dest="log_path", default="log")
    parser.add_option("-o", "--modelPath", dest="model_path")

    (options, args) = parser.parse_args(sys.argv)
    embedding_dim = int(options.embedding_dim)
    model_path = options.model_path
    app_name = "stream_classifier"
    sequence_len = 50   #number of words for each text
    max_words = 1000    

    sc = SparkContext(appName=app_name, conf=create_spark_conf())
    sc.setLogLevel("ERROR")
    init_engine()

    if options.action == "train":
        batch_size = int(options.batchSize)
        max_epoch = int(options.max_epoch)
        #p = float(options.p)
        checkpoint_path = options.checkpoint_path
        logdir = options.log_path
        train_file = "train.txt"
        training_split = 0.8
        class_num = 2 

        print('loading training data...')
        #get training data
        (train_data,nlines) = get_train_data(train_file)
        #train_data = get_data('./acllmdb','train')
        print('train data loaded!')
        print('number of lines for training set: ' + str(nlines))

        #obtain trained model
        train_model = train(sc,
                            train_data,
                            class_num,
                            batch_size,
                            sequence_len, 
                            max_words, 
                            embedding_dim, training_split, checkpoint_path, logdir, app_name)

        sc.stop()

    elif options.action == "streaming_test":
        ssc = StreamingContext(sc, 5)
        topic = "sentiment"  #kafka topic
        zkQuorum = 'localhost:2181' #zk server

        #get the trained model from the model path
        train_model = Model.load(model_path)

        #generate and handle stream
        kafkastream = KafkaUtils.createStream(ssc, zkQuorum, "spark-streaming-consumer", {topic: 1})
        ks = kafkastream.map(lambda ks: (ks[1],0)) #get lines with fake classes
        ks.foreachRDD(show) #show lines
        featstream = ks.transform(lambda rdd: 
                                    preprocess_texts(rdd, sequence_len, max_words, embedding_dim, True)) #get w2v
        featstream.foreachRDD(lambda fs: classify_stream(fs,train_model)) #classify stream

        ssc.start()
        ssc.awaitTermination()

    elif options.action == "test":
        test_data = get_test_data("test.txt")
        test_data_rdd = sc.parallelize(test_data, 2)
        sample_rdd = preprocess_texts(test_data_rdd, sequence_len, max_words, embedding_dim, False)
        #get the trained model from the model path
        train_model = Model.load(model_path)
        predictions = train_model.predict(sample_rdd).take(50)
        #get max probability indices
        y_pred = np.array([ map_predict_label(s) for s in predictions])
        i = 0
        for y in y_pred:
            print(test_data[i][0], end=' ')
            if y==0: 
                print('(NEGATIVE)')
            else:
                print('(POSITIVE)')
            print('\n')
            i = i+1
