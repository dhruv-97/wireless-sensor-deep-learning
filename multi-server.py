import socket
import sys
from threading import Thread, Event
import time

# Multithreaded Python server : TCP Server Socket Program Stub
PORT = 6572
BUFFER_SIZE = 20  # Usually 1024, but we need quick response
NODES = 72
RETAIN = 1000
SLEEP = 0.2
event = Event()

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
            while True:
                data = self.conn.recv(BUFFER_SIZE).decode("utf-8")
                TS, feature = [int(x) for x in data.split(',')]
                self.server.clientTS = max(self.server.clientTS, TS) if TS else 0
                self.server.features[TS][self.clientNum] = feature
                if data == 'exit':
                    break
            self.conn.close()


    class ClearThread(Thread):

        def __init__(self, features):
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

    class SendTSReply(Thread):
        
        def __init__(self, client):
            self.server = server

        def run(self):
            while True:
                time.sleep(SLEEP)
                for i,x in enumerate(self.server.features[self.server.serverTS]:
                    if not x:
                        self.server.threads[i].conn.send(str(self.server.clientTS).encode("utf-8"))
           
    def __init__(self):
        self.features = [[None] * NODES for _ in range(RETAIN)]
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
        reply_thread = Server.SendTSReply(self)
        reply_thread.start()
        for t in self.threads:
            t.start()
        for _ in range(2000):
            time.sleep(SLEEP)
            row = self.serverTS
            print(self.features[row])
            self.serverTS += 1
            if self.serverTS == RETAIN:
                self.serverTS = 0
                event.set()
            elif self.serverTS == RETAIN//2:
                event.set()
        for t in self.threads:
            t.join()
        reply_thread.join()
        clear_thread.join()
        self.soc.close()


S = Server()
S.start()
