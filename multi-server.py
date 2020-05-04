import socket
from threading import Thread, Event
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Saving the minimum, maximum and standard deviation values for normalizing the streaming data in the range [0,1]
test = pd.read_csv("./test.txt")
test_X = np.array(test.iloc[:, 1:])
scaler = MinMaxScaler()
scaler.fit(test_X)

# Multithreaded Python server : TCP Server Socket Program

PORT = 6573 # The port where the client and server would interact
BUFFER_SIZE = 20  # Usually 1024, but we need quick response
NODES = 73 # The number of clients
RETAIN = 1000 # How many rows of recent data to keep 
SLEEP = 1 # The frequency of processing the data
ALPHA = 0.7 # Weightage give to the recent data when esitmating the next value

clearEvent = Event() #Event to indicate the Clearning Thread
queryEvent = Event() #Event to indicate the Query Processor

#Loading the CNN Model for Deep Learning Analysis
model = tf.keras.models.load_model('./saved_model/cnn-model-2')
# Multithreaded Python server : TCP Server Socket Thread Pool

class Server:

    # NODES # of instances of this thread
    class ClientThread(Thread):

        def __init__(self, server, conn, ip, port):
            Thread.__init__(self)
            self.server = server
            self.conn = conn
            self.ip = ip
            self.port = port

            # The first message received from any client is it's clientNum
            self.clientNum = int(conn.recv(2).decode("utf-8"))
            print("New server socket thread started for ", ip, ":", port, ":", self.clientNum)

        def run(self):
            self.conn.send("0".encode("utf-8"))
            BETA = 1 - ALPHA
            while True:
                data = self.conn.recv(BUFFER_SIZE).decode("utf-8")

                #Every message from the client if of the form TS, feature
                TS, feature = [int(x) for x in data.split(',')]

                #Last message from the client
                if TS == -1:
                    break

                #Server always keeps track of the maximum TS value received from any client
                self.server.clientTS = max(self.server.clientTS, TS) if TS else 0

                #Storing the feature in the appropriate row and column
                self.server.features[TS][self.clientNum] = feature

                #Estimating the next value using EWMA
                self.server.predicted[(TS+1)%RETAIN][self.clientNum] = int(feature * ALPHA + self.server.predicted[TS][self.clientNum] * BETA)
            self.conn.close()

    # Thread to clear 50% of data matrix
    class ClearThread(Thread):

        def __init__(self, features):
            Thread.__init__(self)

            #Initially the start and end positions are at 0 and 50%
            self.s = 0
            self.e = RETAIN//2
            self.features = features

        def run(self):
            while True:

                # Waiting to reach either second half or first half
                clearEvent.wait()

                #Clear the other half
                for i in range(self.s, self.e):
                    for j in range(NODES):
                        self.features[i][j] = None

                # Get ready to clear the other half by setting the start and end positions appropriately
                self.swap()
                clearEvent.clear()

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
            for _ in range(1000):

                # Waiting for the main thread which triggers every second
                queryEvent.wait()

                # Release time for query generetor and query processing
                time.sleep(SLEEP/2)
                # Checking for missing values in the row currently being processed
                for i,x in enumerate(self.server.features[self.server.serverTS]):
                    if not x:

                        # Replacing missing value with the estimated value
                        self.server.predicted[(self.server.serverTS+1)%RETAIN][i] = self.server.features[self.server.serverTS][i] = self.server.predicted[self.server.serverTS][i]

                        # Sending the client the maximum value of TS reached on server so far
                        self.server.threads[i].conn.send(str(self.server.clientTS).encode("utf-8"))

                self.total += 1

                # Converting the row into a numpy array
                X_instance = np.array([self.server.features[self.server.serverTS][:NODES-1]]).reshape(1,72).astype(float)

                # Normalizing
                X_normal = scaler.transform(X_instance)

                # Replacing NAN values
                X_normal = np.nan_to_num(X_normal)

                # Reshaping into an 8 x 9 matrix
                X_normal = X_normal.reshape(1,8,9)

                # Passing into the model
                predictedY = model.predict(X_normal)

                predicted, actual = np.argmax(predictedY[0]), self.server.features[self.server.serverTS][NODES-1]//3
                if predicted == actual:
                    self.correct += 1
                print(self.server.features[self.server.serverTS])
                print("Predicted = ", predicted , ", Actual = ", actual)
                queryEvent.clear()

        # Computing the total accuracy in the end
        def totalAccuracy(self):
            return self.total/self.correct

    def __init__(self):

        # RETAIN x NODES matrix for features as well estimated values
        self.features = [[0] * NODES for _ in range(RETAIN)]
        self.predicted = [[0] * NODES for _ in range(RETAIN)]
        self.threads = []
        # Using TCP connection
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.soc.bind((socket.gethostname(), PORT))
        self.clientTS = 0 # Keeping track of the maximum client TS received so far
        self.serverTS = 0 # Keeping track of the current row being processes

    def start(self):
        # Waiting to establish connections with 73 clients
        while len(self.threads) < NODES:
            self.soc.listen(NODES)
            print("Multithreaded Python server : Waiting for connections from TCP clients...")
            # Accepting an indiviual socket connection with a client
            (conn, (ip, port)) = self.soc.accept()
            # Creating a thread for each socket connection
            self.threads.append(Server.ClientThread(self, conn, ip, port))

        # Starting the Clearing Thread
        clear_thread = Server.ClearThread(self.features)
        clear_thread.start()

        # Starting the Query Generetor Thread
        query_generator = Server.QueryGenerator(self)
        query_generator.start()

        # Server has established connection with all clients. Now will send timestamps
        for t in self.threads:
            t.start()

        # Do query processing every second for 1000 test cases
        for _ in range(1000):
            time.sleep(SLEEP)

            # Setting the event to do query processing
            queryEvent.set()

            # Get ready to process the next row
            self.serverTS += 1

            # Check to see if we have reached the first or second half so that we can activate Clearing Thread
            if self.serverTS == RETAIN:
                self.serverTS = 0
                clearEvent.set()
            elif self.serverTS == RETAIN//2:
                clearEvent.set()

        print(query_generator.totalAccuracy())
        for t in self.threads:
            t.join()
        query_generator.join()
        clear_thread.join()
        self.soc.close()


S = Server()
S.start()
