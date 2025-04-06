import sys
import asyncio
import pickle
import zlib
import pandas as pd
import numpy as np
from phe import paillier
import ShamirSecret
import DLClient

# ======= Client Parameters =======
#HOST = socket.gethostbyname(socket.gethostname()) # gets host local ip
HOST = input("Enter Serverip eg 10.10.138.6")
PORT = 65432
client_id = int(sys.argv[1])

# ======= Generate Paillier Key Pair =======
public_key, secret_key = paillier.generate_paillier_keypair()

# ======= Server Parameters =======
THRESHOLD = None
BIG_P = None
ACTIVITY = 'run'

final_weights = None
dataset = None

# ======= Initialize =======
async def initialize():
    global dataset
    reader, writer = await asyncio.open_connection(HOST, PORT)

    # Send public key to server
    data = {
        'client_id': client_id,
        'public_key': public_key
    }

    dataset = pd.read_csv(f"s{client_id}_{ACTIVITY}_processed.csv")
    print(f"Client {client_id}'s data : ", dataset.shape)

    compressed_data = zlib.compress(pickle.dumps(data))
    writer.write(compressed_data)
    await writer.drain()

    global THRESHOLD, BIG_P

    # Receive parameters
    data = await reader.read(4096)
    data = pickle.loads(zlib.decompress(data))
    THRESHOLD = data['threshold']
    BIG_P = data['p']

    round_no = 1
    while True:
        if not await aggregate_weight(reader, writer, round_no):
            break
        print(f"[CLIENT {client_id}] Round {round_no} has been completed. Proceeding to the next round ...")
        round_no += 1

    print(f"[CLIENT {client_id}] All the rounds has been completed")
    print(f"Terminating CLIENT {client_id} !!!")
    writer.close()
    await writer.wait_closed()


async def aggregate_weight(reader, writer, round_no):
    global final_weights

    data = {'alive': False if dataset.shape[0] < (round_no - 1) * 2**15 else True}
    compressed_data = zlib.compress(pickle.dumps(data))
    writer.write(compressed_data)
    await writer.drain()

    if not data['alive']:
        return False

    print(f"[CLIENT {client_id}] Round {round_no} has been started.")

    # ======= Generate Local Model =======
    if round_no == 1:
        local_weights = DLClient.modelTraining(dataset[:2**15])
    else:
        local_weights = DLClient.modelTraining(dataset[(round_no - 1)*2**15 : min(dataset.shape[0], (round_no)*2**15)], final_weights)
    

    # Receive public keys of all clients in the round
    data = await reader.read(2**20)
    public_keys = pickle.loads(zlib.decompress(data))
    
    print(f"[CLIENT {client_id}] Recieved Public Keys from {list(public_keys.keys())}")


    # ======= Generate masking value =======
    shamir_secret = np.random.randint(1, BIG_P - 1)
    print(f"[CLIENT {client_id}] Round {round_no} Shamir Secret Key: {shamir_secret}")
    
    # ======= Generate Shamir Secret Shares =======
    shares = ShamirSecret.generate_share(shamir_secret, list(public_keys.keys()), THRESHOLD)

    # ======= Encrypt Weights =======
    ciphertext = [w + shamir_secret for w in local_weights]

    # Encrypt and store shares
    encrypted_shares = {
        cid: public_keys[cid].encrypt(shares[cid])
        for cid in public_keys
    }

    # Compress and send encrypted ciphertext and shares to server
    data = {
        'ciphertext': ciphertext,
        'shares': encrypted_shares,
        'round_no': round_no
    }
    compressed_data = zlib.compress(pickle.dumps(data))
    writer.write(compressed_data)
    await writer.drain()
    print(f"[CLIENT {client_id}] Round {round_no} : Sent ciphertext and encrypted shares to server")


    # Receive Aggregated Share from Server
    data = await reader.read(4096)
    if not data:
        print(f"[CLIENT {client_id}] Round {round_no} : Couldn't receive aggregated share from the server")
        return False
    
    aggregated_share = pickle.loads(zlib.decompress(data))

    # Decrypt aggregated share using secret key
    decrypted_share = secret_key.decrypt(aggregated_share)
    
    # Send back encrypted share to server
    compressed_data = zlib.compress(pickle.dumps(decrypted_share))
    writer.write(compressed_data)
    await writer.drain()
    print(f"[CLIENT {client_id}] Round {round_no} : Sent decrypted share to server")


    # ======= Receive Final Aggregated Global Model =======
    data = await reader.read(2**20)
    if not data:
        print(f"[CLIENT {client_id}] Round {round_no} : Couldn't receive final aggregated global model from the server")
        return False
    data = pickle.loads(zlib.decompress(data))

    aggregated_shamir_secret = secret_key.decrypt(data['aggregated_shamir_secret'])
    aggregated_ciphertext = data['aggregated_ciphertext']
    n = data['n']

    # Compute final model after unmasking
    final_weights = [(ct - aggregated_shamir_secret) / n for ct in aggregated_ciphertext]

    print(f"[CLIENT {client_id}] Round {round_no} : Training and aggregation complete.")
    
    return True


# Start client connection
asyncio.run(initialize())
