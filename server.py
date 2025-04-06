import asyncio
import pickle
import zlib
import numpy as np
from phe import paillier
import ShamirSecret, socket

# ======= Server Parameters =======
HOST = '0.0.0.0'
PORT = 65432
THRESHOLD = 1
BIG_P = 104729  # Large prime for modular operation

# Shared Variable
public_keys = {}
clients = set()
ciphertexts = {}
ciphertext_list = []
encrypted_shares = {}
aggregated_ciphertext = None
aggregated_shares = {}
decrypted_shares = {}
aggregated_shamir_secret = None
count = 0

lock = asyncio.Lock()
semaphore = asyncio.Semaphore(THRESHOLD)


# ======= Handle Client Connection =======
async def handle_client(reader, writer):
    global public_keys
    
    # Receive public key from client
    data = await reader.read(4096)
    data = pickle.loads(zlib.decompress(data))
    client_id = data['client_id']
    public_keys[client_id] = data['public_key']

    print(f"[SERVER] Received public key from Client {client_id}")

    # Once all keys are received, send parameters to all clients
    response = {
        'threshold': THRESHOLD,
        'p': BIG_P
    }
    compressed_response = zlib.compress(pickle.dumps(response))
    writer.write(compressed_response)
    await writer.drain()
    print(f"[SERVER] Sent parameters to Client {client_id}")

    while True:  # Keep aggregating for multiple rounds
        if not await aggregate_weight(reader, writer, client_id):
            break  # Exit loop when clients signal training completion
    
    print(f"[SERVER] All the rounds has been completed fot client {client_id}!!!")
    

async def aggregate_weight(reader, writer, client_id):
    global clients, ciphertexts, encrypted_shares, aggregated_ciphertext, aggregated_shares, decrypted_shares, aggregated_shamir_secret, count

    # Keep alive Signal
    data = await reader.read(1028)
    data = pickle.loads(zlib.decompress(data))
    if not data.get("alive"):
        return False  # Stop if client sends a termination signal

    async with semaphore:
        # Critical Section
        # Select clients for the round
        async with lock:
            clients.add(client_id)
        print(f"[SERVER] Client {client_id} is selected for the round")

        while len(clients) < THRESHOLD:
            print(f"[SERVER] Waiting for all clients to join the round... ({len(clients)}/{THRESHOLD})")
            await asyncio.sleep(1)  # Avoid busy waiting
        print(clients)

        # Once all clients are join, send public keys to all clients in the round
        response = {client: public_keys[client] for client in clients}
        compressed_response = zlib.compress(pickle.dumps(response))
        writer.write(compressed_response)
        await writer.drain()
        print(f"[SERVER] Sent public keys to Client {client_id}")


        # Receive encrypted ciphertext and shares from client
        data = await reader.read(2**20)
        if not data:
            print(f"[SERVER] No data received from Client {client_id}")
            return False  # Stop if no data received
        
        data = pickle.loads(zlib.decompress(data))

        ciphertexts[client_id] = data['ciphertext']
        encrypted_shares[client_id] = data['shares']

        async with lock:
            count += 1

        print(f"[SERVER] Received ciphertext and shares from Client {client_id}")


        # If all clients have submitted weights and shares, aggregate them
        if count == THRESHOLD:
            
            # Aggregate encrypted local models homomorphically
            ciphertext_list = list(ciphertexts.values())
            aggregated_ciphertext = [np.sum(np.array(layer_ciphertexts), axis=0) for layer_ciphertexts in zip(*ciphertext_list)]
            
            print(f"[SERVER] Aggregated Ciphertext: {aggregated_ciphertext}")

            # Aggregate Shamir shares
            aggregated_shares = encrypted_shares[list(ciphertexts.keys())[0]].copy()
            for i in list(ciphertexts.keys())[1:]:
                for j in ciphertexts:
                    aggregated_shares[j] = aggregated_shares[j] + encrypted_shares[i][j]      # Homomorphic Addition            

            count = 0

        while count > 0:
            print(f"[SERVER] Waiting for all clients to send weights... ({len(encrypted_shares)}/{THRESHOLD})")
            await asyncio.sleep(1)  # Avoid busy waiting


        # Send aggregated shares to client
        compressed_data = zlib.compress(pickle.dumps(aggregated_shares[client_id]))
        writer.write(compressed_data)
        await writer.drain()
        print(f"[SERVER] Sent aggregated share to Client {client_id}")


        # Receive decrypted shares from clients
        data = await reader.read(4096)
        data = pickle.loads(zlib.decompress(data))

        # Store the decrypted share
        decrypted_shares[client_id] = data
        
        # ======= Reconstruct Secret Using Shamir =======
        if len(decrypted_shares) == THRESHOLD:
            aggregated_shamir_secret = ShamirSecret.reconstruct_secret(decrypted_shares)
            print(f"[SERVER] Reconstructed Shamir Secret: {aggregated_shamir_secret}")

        while len(decrypted_shares) < THRESHOLD:
            print(f"[SERVER] Waiting for all clients to finish decryption...")
            await asyncio.sleep(1)


        # Send final aggregated global model
        final_model = {
            'aggregated_shamir_secret': public_keys[client_id].encrypt(aggregated_shamir_secret),
            'aggregated_ciphertext': aggregated_ciphertext,
            'n': THRESHOLD
        }

        compressed_data = zlib.compress(pickle.dumps(final_model))
        writer.write(compressed_data)
        await writer.drain()
        print(f"[SERVER] Sent Final Model to Client {client_id}")
        
        async with lock:
            count += 1

        if count == THRESHOLD:
            clients.clear()
            ciphertexts.clear()
            encrypted_shares.clear()
            aggregated_ciphertext = None
            aggregated_shares.clear()
            decrypted_shares.clear()
            aggregated_shamir_secret = None
            count = 0
            print(f"[SERVER] Waiting for the next round of aggregation ... ")

        while count > 0:
            print(f"[SERVER] Waiting for all clients to finish the round...")
            await asyncio.sleep(1)

    return True
        

# ======= Start Server =======
async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"[SERVER] Server started on {addr}")

    async with server:
        await server.serve_forever()

# Run server
asyncio.run(main())
