import sys
sys.path.append('/Users/yaojiahao/anaconda3/envs/myenv/lib/python3.11/site-packages')
import sys
import socket
import io
import os
import dill
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from fl_system import receive_data
from model import LeNet
from fl_system import single_train
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_private_data(client_id, param):
    with open(os.path.join(param.data_dir, f"Client{client_id}.pkl"), "rb") as f:
        private_data = dill.load(f)
        private_dataloader = DataLoader(
            private_data,
            batch_size=param.batch_size,
            shuffle=True,
            drop_last=True
        )
    return private_dataloader

def connect_to_server(param):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((param.server_address, int(param.port)))
    return client_socket

def receive_and_update_model(client_socket, client_model, param, private_dataloader):
    while True:
        try:
            # receive global state
            data = receive_data(client_socket, param.buffer_size)
            if data == b"FIN":
                logger.info(f"Received end signal. Closing client.")
                break
            elif data is None:
                logger.warning("No data received, connection may be closed.")
                break
            
            buffer = io.BytesIO(data)
            buffer.seek(0)
            global_model_state = torch.load(buffer)
            
            client_model.load_state_dict(global_model_state)
            single_train(client_model, param.lr, private_dataloader, param.n_epochs, param.device)

            # send client state
            buffer = io.BytesIO()
            torch.save(client_model.state_dict(), buffer)
            buffer.seek(0)
            client_socket.sendall(buffer.getvalue())
            client_socket.sendall(b"END")
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            break

def client_process(client_id, param):
    private_dataloader = load_private_data(client_id, param)
    client_socket = connect_to_server(param)
    client_model = LeNet(param.input_size, param.output_channel).to(param.device)

    receive_and_update_model(client_socket, client_model, param, private_dataloader)

    client_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <param_file>")
        sys.exit(1)
    
    client_id, param_file = sys.argv[1:3]
    param = OmegaConf.load(param_file)

    client_process(client_id, param)
