import socket
import sys
from threading import Thread
import time

# Multithreaded Python server : TCP Server Socket Program Stub
PORT = int(sys.argv[1])
BUFFER_SIZE = 20  # Usually 1024, but we need quick response
NODES = 72


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
                self.server.clientTS = max(self.server.clientTS, TS)
                self.server.features[TS%1000][self.clientNum] = feature
                if data == 'exit':
                    break
            self.conn.close()



    def __init__(self):
        self.features = [[None] * NODES for _ in range(1000)]
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
        for t in self.threads:
            t.start()
        for _ in range(2000):
            time.sleep(0.2)
            row = self.serverTS%1000
            print(self.features[row])
            for i,x in enumerate(self.features[row]):
                if not x:
                    print(self.clientTS)
                    self.threads[i].conn.send(str(self.clientTS).encode("utf-8"))
            self.serverTS += 1
        for t in self.threads:
            t.join()

        self.soc.close()


S = Server()
S.start()
