# Deep distributed sentiment analysis on data stream (example)
Text <b>stream</b> sentiment analysis using a <b>distributed</b> deep learning approach based on Convolutional Neural Networks. 

Language: Python 2.7

Streaming platform: [Kafka](https://kafka.apache.org/)

Distributed deep learning: [Spark](https://spark.apache.org/) + [BigDL](https://bigdl-project.github.io)

# Authors

Mario Ruggieri

e-mail: mario.ruggieri@uniparthenope.it

# Dependencies

- Intel BigDL with Spark: https://bigdl-project.github.io/master/#PythonUserGuide/install-from-pip/
- Kafka 0.8.2.2: https://kafka.apache.org/downloads

# Usage

**Starting Kafka ecosystem (bin is in kafka_2.11-0.8.2.2):**

  ./bin/zookeeper-server-start.sh config/zookeeper.properties
  ./bin/kafka-server-start.sh config/server.properties

**Starting producers:**

  python [HERE PATH TO kafka_producer.py]
  python [HERE PATH TO kafka_producer_for_word_prediction.py]

**Stream testing with pretrained models:**

    ${SPARK_HOME}/bin/spark-submit  \
    --py-files ${PYTHON_API_ZIP_PATH},[HERE ABSOLUTE PATH TO cnn_stream_classifier.py]  \
    --jars ${BigDL_JAR_PATH}  \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar  \
    --conf spark.executorEnv.PYTHONHASHSEED=${PYTHONHASHSEED} \
    --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 \
    --driver-memory 10g  \
    --executor-cores 4 \
    --executor-memory 60g \
    --num-executors 1 \
    [HERE RELATIVE PATH TO cnn_stream_classifier.py] \
    --action streaming_test \
    --modelPath [HERE PATH TO model_for_sentiment]

    ${SPARK_HOME}/bin/spark-submit  \
    --py-files ${PYTHON_API_ZIP_PATH},[HERE ABSOLUTE PATH TO lstm_word_prediction_single_out.py]  \
    --jars ${BigDL_JAR_PATH}  \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar  \
    --conf spark.executorEnv.PYTHONHASHSEED=${PYTHONHASHSEED} \
    --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 \
    --driver-memory 10g  \
    --executor-cores 4 \
    --executor-memory 60g \
    --num-executors 1 \
    [HERE RELATIVE PATH TO lstm_word_prediction_single_out.py]  \
    --action streaming_test \
    --modelPath [HERE PATH TO model_for_word_pred]

**Training (generating models):**

    ${SPARK_HOME}/bin/spark-submit  \
      --py-files ${PYTHON_API_ZIP_PATH},[HERE ABSOLUTE PATH TO cnn_stream_classifier.py] \
      --jars ${BigDL_JAR_PATH}  \
      --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
      --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar  \
      --conf spark.executorEnv.PYTHONHASHSEED=${PYTHONHASHSEED} \
      --driver-memory 10g  \
      --executor-cores 4 \
      --executor-memory 60g \
      --num-executors 1 \
      [HERE RELATIVE PATH TO cnn_stream_classifier.py] \
      --checkpoint_path [HERE PATH TO THE CHECKPOINT PATH] \
      --log_path [HERE PATH TO THE LOG PATH]

    ${SPARK_HOME}/bin/spark-submit  \
    --py-files ${PYTHON_API_ZIP_PATH},[HERE ABSOLUTE PATH TO lstm_word_prediction_single_out.p] \
    --jars ${BigDL_JAR_PATH}  \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar  \
    --conf spark.executorEnv.PYTHONHASHSEED=${PYTHONHASHSEED} \
    --driver-memory 10g  \
    --executor-cores 4 \
    --executor-memory 60g \
    --num-executors 1 \
    [HERE RELATIVE PATH TO lstm_word_prediction_single_out.p] \
    --checkpoint_path [HERE PATH TO THE CHECKPOINT PATH]
    --log_path [HERE PATH TO THE LOG PATH]

# License

Please read <b>Apache 2.0</b> License file


