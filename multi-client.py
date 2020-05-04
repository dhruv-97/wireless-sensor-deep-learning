# Python TCP Client
import socket
import sys

import time
import threading

NODES = 73 # Number of clients

b = threading.Barrier(NODES + 1)
host = socket.gethostbyname(sys.argv[1]) # take server name as command line arguement
port = 6573 # Port on which the clients and server interact
BUFFER_SIZE = 20

RETAIN = 1000 # When to reset the TS

class Client:
    class ClientThread(threading.Thread):

        def __init__(self, client, clientNum):
            threading.Thread.__init__(self)
            
            # Establishing socket connection with the server
            self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client = client
            self.clientNum = clientNum
            self.TS = 0
            self.soc.connect((host, port))

            # First message sent to server is always the clientNum
            self.soc.send(str(clientNum).encode("utf-8"))

        def run(self):
            # First message received from the server is the TS to start with
            self.TS = int(self.soc.recv(BUFFER_SIZE))
            time.sleep(0.1)
            while True:
                # Take the value with the column index as clientNum
                if self.client.state[self.clientNum]:
                    # Send TS, feature to client
                    data = str(self.TS) + ',' + str(self.client.data[self.clientNum])
                    self.soc.send(data.encode("utf-8"))

                    # Increase the TS
                    self.TS += 1

                    # Reseting the TS value
                    if self.TS == RETAIN:
                        self.TS = 0
                    b.wait()

                # Simulating the crashing of nodes
                else:
                    # If the state of any client is False, it means it is currently not sending any data
                    while not self.client.state[self.clientNum]:
                        self.soc.recv(BUFFER_SIZE)
                        b.wait()
                    # Broke out of the loop => state of this client has been reset to True 
                    # Client receives the maximum value of TS the server has seen from any client when it recovers
                    self.TS  = int(self.soc.recv(BUFFER_SIZE).decode("utf-8"))
                if self.client.exit:
                    self.soc.send("-1,-1".encode("utf-8"))
                    break

            self.soc.close()

    # To simulate crashing of the sensor nodes
    class CommandThread(threading.Thread):

        def __init__(self, client):
            threading.Thread.__init__(self)
            self.client = client

        def run(self):
            while not self.client.exit:
                cmd = input("comp512>")
                args = cmd.split(" ")
                # Setting the state as False so that this client cannot send data
                if 'pause' == args[0]:
                    self.client.state[int(args[1])] = False
                # Setting the state as True so that it can keep sending the data again after receiving TS from the server
                elif 'start' == args[0]:
                    self.client.state[int(args[1])] = True
                elif 'exit' == args[0]:
                    self.client.exit = True
                else:
                    print("Invalid command")

    def __init__(self, NODES):
        self.NODES = NODES
        self.state = [True] * NODES
        self.data = None
        self.threads = []
        self.exit = False

    def start(self):
        # Creating a thread for each client
        for i in range(self.NODES):
            self.threads.append(Client.ClientThread(self, i))

        # Starting the threads
        for thread in self.threads:
            thread.start()

        # Starting the Command Thread
        command_thread = Client.CommandThread(self)
        command_thread.start()

        # Opening the data file
        f = open("./test.txt")

        # Reading the data file line by line every second while the client threads take their respective values
        for j, line in enumerate(f):
            data = [x for x in line.split(",")]
            self.data = data[1:] + [data[0]]
            b.wait()
            if self.exit:
                break

            # Updating data every second. Clients can only send the updated values because of Barrier synchronization
            time.sleep(1)

        self.exit = True
        command_thread.join()
        for thread in self.threads:
            thread.join()


C = Client(NODES)
C.start()
