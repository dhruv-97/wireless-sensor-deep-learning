import socket
from threading import Thread, Event
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
test = pd.read_csv("./test.txt")
test_X = np.array(test.iloc[:, 1:])
scaler = MinMaxScaler()
scaler.fit(test_X)

# Multithreaded Python server : TCP Server Socket Program Stub
PORT = 6572
BUFFER_SIZE = 20  # Usually 1024, but we need quick response
NODES = 73
RETAIN = 1000
SLEEP = 1
ALPHA = 0.7
event = Event()


model = tf.keras.models.load_model('saved_model/cnn-model-2')
# Multithreaded Python server : TCP Server Socket Thread Pool

class Server:
    class ClientThread(Thread):

        def __init__(self, server, conn, ip, port):
            Thread.__init__(self)
            self.server = server
            self.conn = conn
            self.ip = ip
            self.port = port
            self.clientNum = int(conn.recv(2).decode("utf-8"))
            print("New server socket thread started for ", ip, ":", port, ":", self.clientNum)

        def run(self):
            self.conn.send("go".encode("utf-8"))
            BETA = 1 - ALPHA
            while True:
                data = self.conn.recv(BUFFER_SIZE).decode("utf-8")
                TS, feature = [int(x) for x in data.split(',')]
                if TS == -1:
                    break
                self.server.clientTS = max(self.server.clientTS, TS) if TS else 0
                self.server.features[TS][self.clientNum] = feature
                if feature:
                    self.server.predicted[(TS+1)%RETAIN][self.clientNum] = int(feature * ALPHA + self.server.predicted[TS][self.clientNum] * BETA)
            self.conn.close()


    class ClearThread(Thread):

        def __init__(self, features):
            Thread.__init__(self)
            self.s = 0
            self.e = RETAIN//2
            self.features = features

        def run(self):
            while True:
                event.wait() 
                for i in range(self.s, self.e):
                    for j in range(NODES):
                        self.features[i][j] = None
                self.swap()
                event.clear()

        def swap(self):
            if self.s:
                self.s = self.e
                self.e = RETAIN
            else:
                self.e  = self.s
                self.s = 0

    class QueryGenerator(Thread):
        
        def __init__(self, server):
            Thread.__init__(self)
            self.server = server
            self.total = 0
            self.correct = 0

        def run(self):
            time.sleep(SLEEP/2)
            for _ in range(1000):
                time.sleep(SLEEP)
                for i,x in enumerate(self.server.features[self.server.serverTS]):
                    if not x:
                        self.server.predicted[(self.server.serverTS+1)%RETAIN][i] = self.server.features[self.server.serverTS][i] = self.server.predicted[self.server.serverTS][i]
                        self.server.threads[i].conn.send(str(self.server.clientTS).encode("utf-8"))
                self.total += 1
                X_instance = np.array([self.server.features[self.server.serverTS][:NODES-1]]).reshape(1,72).astype(float)
                X_normal = scaler.transform(X_instance)
                X_normal = np.nan_to_num(X_normal)
                predictedY = model.predict(X_normal)
                predicted, actual = np.argmax(predictedY[0]), self.server.features[self.server.serverTS][NODES-1]//3
                if predicted == actual:
                    self.correct += 1
                print(self.server.features[self.server.serverTS])
                print("Predicted = ", predicted , ", Actual = ", actual)

        def totalAccuracy(self):
            return self.total/self.correct

    def __init__(self):
        self.features = [[0] * NODES for _ in range(RETAIN)]
        self.predicted = [[0] * NODES for _ in range(RETAIN)]
        self.state = [True] * NODES
        self.threads = []
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.soc.bind((socket.gethostname(), PORT))
        self.clientTS = 0
        self.serverTS = 0

    def start(self):
        while len(self.threads) < NODES:
            self.soc.listen(NODES)
            print("Multithreaded Python server : Waiting for connections from TCP clients...")
            (conn, (ip, port)) = self.soc.accept()
            self.threads.append(Server.ClientThread(self, conn, ip, port))
        self.threads.sort(key=lambda x: x.clientNum)
        clear_thread = Server.ClearThread(self.features)
        clear_thread.start()
        query_generator = Server.QueryGenerator(self)
        query_generator.start()
        for t in self.threads:
            t.start()
        for _ in range(1000):
            time.sleep(SLEEP)
            self.serverTS += 1
            if self.serverTS == RETAIN:
                self.serverTS = 0
                event.set()
            elif self.serverTS == RETAIN//2:
                event.set()
        print(query_generator.totalAccuracy())
        for t in self.threads:
            t.join()
        query_generator.join()
        clear_thread.join()
        self.soc.close()


S = Server()
S.start()
