# Python TCP Client
import socket
import sys

import time
import threading

NODES = 72
b = threading.Barrier(NODES + 1)
host = socket.gethostbyname(sys.argv[1])
port = 6572
BUFFER_SIZE = 20
RETAIN = 100

class Client:
    class ClientThread(threading.Thread):

        def __init__(self, client, clientNum):
            threading.Thread.__init__(self)
            self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client = client
            self.clientNum = clientNum
            self.TS = 0
            self.soc.connect((host, port))
            self.soc.send(str(clientNum).encode("utf-8"))

        def run(self):
            self.soc.recv(BUFFER_SIZE)
            time.sleep(0.1)
            while True:
                if self.client.state[self.clientNum]:
                    data = str(self.TS) + ',' + str(self.client.data[self.clientNum])
                    self.soc.send(data.encode("utf-8"))
                    self.TS += 1
                    if self.TS == RETAIN:
                        self.TS = 0
                    b.wait()
                else:
                    while not self.client.state[self.clientNum]:
                        self.soc.recv(BUFFER_SIZE)
                        b.wait()
                    self.TS  = int(self.soc.recv(BUFFER_SIZE).decode("utf-8"))
                if self.client.exit:
                    self.soc.send("exit".encode("utf-8"))
                    break

            self.soc.close()

    class CommandThread(threading.Thread):

        def __init__(self, client):
            threading.Thread.__init__(self)
            self.client = client

        def run(self):
            while not self.client.exit:
                cmd = input("comp512>")
                args = cmd.split(" ")
                if 'pause' == args[0]:
                    self.client.state[int(args[1])] = False
                elif 'start' == args[0]:
                    self.client.state[int(args[1])] = True
                elif 'exit' == args[0]:
                    self.client.exit = True
                else:
                    print("Invalid command")

    def __init__(self, NODES):
        self.NODES = NODES
        self.state = [True] * NODES
        self.row = None
        self.data = None
        self.threads = []
        self.exit = False

    def start(self):
        for i in range(self.NODES):
            self.threads.append(Client.ClientThread(self, i))

        for thread in self.threads:
            thread.start()
        command_thread = Client.CommandThread(self)
        command_thread.start()
        f = open(
            "./WTD_upload/Toluene_200/L1/201106121329_board_setPoint_400V_fan_setPoint_000_mfc_setPoint_Toluene_200ppm_p1")
        for j, line in enumerate(f):

            self.row = j
            data = line.split("\t")
            self.data = [data[i] for i in range(12, 92) if data[i] != '1']
            b.wait()
            if self.exit:
                break
            time.sleep(1)

        command_thread.join()
        for thread in self.threads:
            thread.join()


C = Client(NODES)
C.start()
