import sys
sys.path.append('/Users/yaojiahao/anaconda3/envs/myenv/lib/python3.11/site-packages')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from abc import ABC, abstractmethod
import subprocess, socket, threading, io, copy, os, shutil

import model
from dataset import Dataset

class FL_sys(ABC):
    def __init__(self, param):
        self.device = param.device
        self.client_ckp_dir = param.client_ckp_dir
        self.global_ckp_dir = param.global_ckp_dir
        os.makedirs(self.client_ckp_dir)
        os.makedirs(self.global_ckp_dir)

        self.N = param.n_clients
        self.M = param.n_update_clients
        self.input_size = param.input_size
        self.output_channel = param.output_channel
        self.model = model.LeNet(self.input_size, self.output_channel).to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.batch_size = param.batch_size
        self.n_rounds = param.n_rounds
        self.n_epochs = param.n_epochs
        self.lr = param.lr
        self.test_dataloader = None 
    
    @abstractmethod
    def send_and_train(self, idx, global_state):
        raise NotImplementedError
    
    @abstractmethod
    def aggregate(self, idx):
        raise NotImplementedError
    
    def train(self):
        best_acc = 0
        global_ckp_path = os.path.join(self.global_ckp_dir, "global.pth")
        torch.save(self.global_model.state_dict(), global_ckp_path)

        for r in range(self.n_rounds):
            idx = np.random.choice(self.N, self.M, replace=False)
            
            global_state = torch.load(global_ckp_path)
            self.send_and_train(idx, global_state)                

            avg_model = self.aggregate(idx)
            
            self.global_model.load_state_dict(avg_model.state_dict())
            test_loss, accuracy = test(self.global_model, self.test_dataloader, self.device)
            print(f"Round {r+1}, Test Loss: {test_loss}, Accuracy: {accuracy}")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.global_model.state_dict(), global_ckp_path)

class Offline_Pattern(FL_sys):
    def __init__(self, param):
        super().__init__(param)
        self.dataset = Dataset(param.data_dir, self.N)
        self.client_dataloaders = self.dataset.get_train_dataloaders(self.batch_size)
        self.test_dataloader = self.dataset.get_test_dataloader(self.batch_size)

    def send_and_train(self, idx, global_state):
        client_model = copy.deepcopy(self.model)
        
        for i in tqdm(idx):
            dataloader = self.client_dataloaders[i]
            client_model.load_state_dict(global_state)
            single_train(client_model, self.lr, dataloader, self.n_epochs, self.device)
            
            save_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
            torch.save(client_model.state_dict(), save_path)

    def aggregate(self, idx):
        avg_model = copy.deepcopy(self.model)

        for i in idx:
            client_model_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(torch.load(client_model_path))
            for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                if i == idx[0]:
                    avg_param.data = client_param.data
                else:
                    avg_param.data += client_param.data

        for avg_param in avg_model.parameters():
            avg_param.data /= self.M
        
        return avg_model
    
class Online_Pattern(FL_sys):
    def __init__(self, param, config_path):
        super().__init__(param)
        self.dataset = Dataset(param.data_dir, self.N, load_train=False)
        self.test_dataloader = self.dataset.get_test_dataloader(self.batch_size)
        
        self.port = param.port
        self.server_address = param.server_address
        self.buffer_size = param.buffer_size

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        
        try:
            print(f"Attempting to bind to {self.server_address}:{self.port}")
            self.server_socket.bind((self.server_address, self.port))
            self.server_socket.listen(self.N)
            print(f"Successfully bound to {self.server_address}:{self.port}")
        except OSError as e:
            print(f"Error binding to {self.server_address}:{self.port} - {e}")
            raise

        # start client process
        for i in range(self.N):
            subprocess.Popen(["python", "client.py", str(i+1), str(config_path)])
            print(f"Client {i+1} acvitated.")
        
        # connect client
        self.client_sockets = []
        self.lock = threading.Lock()
        for _ in range(self.N):
            client_socket, addr = self.server_socket.accept()
            self.client_sockets.append(client_socket)
            print(f"Connect to client at {addr}")

    def send_and_train(self, idx, global_state):
        threads = []
        for i in idx:
            thread = threading.Thread(target=self.send_model, args=(self.client_sockets[i], global_state, i))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def send_model(self, client_socket, global_state, i):
        try:
            buffer = io.BytesIO()
            torch.save(global_state, buffer)
            buffer.seek(0)
            client_socket.sendall(buffer.getvalue())
            client_socket.sendall(b"END")
            print(f"Send model to client {i}")
        except socket.error as e:
            print("Socket error during sending model:", e)

    def aggregate(self, idx):
        threads = []
        self.avg_model = copy.deepcopy(self.model)
        for avg_param in self.avg_model.parameters():
            avg_param.data.zero_()

        for i in idx:
            thread = threading.Thread(target=self.receive_model, args=(self.client_sockets[i], i, idx))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        for avg_param in self.avg_model.parameters():
            avg_param.data /= self.M
        
        return self.avg_model

    def receive_model(self, client_socket, i, idx):
        try:
            data = receive_data(client_socket, self.buffer_size)
            if data is None:
                print("Socket error during receiving client model")
            
            buffer = io.BytesIO(data)
            buffer.seek(0)
            client_model_state = torch.load(buffer)

            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(client_model_state)
            save_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
            torch.save(client_model.state_dict(), save_path)
            with self.lock:
                for avg_param, client_param in zip(self.avg_model.parameters(), client_model.parameters()):
                    avg_param.data += client_param.data
            print(f"Recieve model from client {i}")
        except socket.error as e:
            print("Socket error during receiving client model:", e)
    
    def train(self):
        super().train()

        for client_socket in self.client_sockets:
            try:
                client_socket.sendall(b"FIN")
                client_socket.close()
            except socket.error as e:
                print(f"Error sending end signal: {e}")

# train at one client
def single_train(model, lr, dataloader, n_epochs, device):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for e in range(n_epochs):
        for features, labels in dataloader:
            features, labels = features.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(features)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, test_dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.squeeze().long().to(device)
            outputs = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(outputs, target)

            predict = outputs.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()
    accuracy = correct / len(test_dataloader.dataset)
    return test_loss, accuracy


def receive_data(sock, buffer_size):
    data = b''
    while True:
        try:
            packet = sock.recv(buffer_size)
            if packet:
                data += packet
                if data[-3:] == b"FIN":  
                    data = b"FIN"
                    break
                if data[-3:] == b"END":
                    data = data[:-3]  
                    break
            else:
                break
        except socket.error as e:
            print("Socket error while receiving data:", e)
            return None

    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the config file")
    args, extras = parser.parse_known_args()

    param = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if param.mode == "offline":
        pipe = Offline_Pattern(param)
        pipe.train()

    if param.mode == "online":
        pipe = Online_Pattern(param, args.config)
        pipe.train()