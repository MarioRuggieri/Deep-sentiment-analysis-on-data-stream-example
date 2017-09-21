from kafka import KafkaProducer
import time
import random

def generate_stream(testFile):
    topic = 'sentiment'
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    '''
    for c in ('pos','neg'):
        path = 'aclImdb/test/%s' % c
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r') as f:
                review = f.read()
                producer.send(topic,review)
    '''

    print 'streaming...'
    with open(test_file) as f:
        for line in f:
    #for line in lines:
            producer.send(topic,line)
            print("line sent")
            time.sleep(10)

if __name__ == '__main__':
    test_file = "test.txt"
    generate_stream(test_file)

