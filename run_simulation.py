import sionna
import numpy as np
import tensorflow as tf
import torch
import math
import scipy.io
#%%
import heapq
from collections import defaultdict

def build_huffman_tree(sentences):
    if not sentences:
        return None

    # Calculate frequency of each word across all sentences
    freq_dict = defaultdict(int)
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            freq_dict[word] += 1

    # Create a priority queue with initial nodes
    priority_queue = [Node(word, freq) for word, freq in freq_dict.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        # Pop the two nodes with the smallest frequency
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        # Create a new internal node with these two nodes as children
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        # Add the new node to the priority queue
        heapq.heappush(priority_queue, merged)

    # The remaining node is the root of the Huffman tree
    return priority_queue[0]



class Node:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
        self.left = None
        self.right = None

    # Define comparators for priority queue
    def __lt__(self, other):
        return self.freq < other.freq

def generate_huffman_codes(root):
    codes = {}
    code_lengths = {}

    def generate_codes_helper(node, current_code):
        if node is None:
            return

        if node.word is not None:
            codes[node.word] = current_code
            code_lengths[node.word] = len(current_code)
            return

        generate_codes_helper(node.left, current_code + [0])
        generate_codes_helper(node.right, current_code + [1])

    generate_codes_helper(root, [])
    return codes, code_lengths


def encode_sentence(sentence, codes):
    encoded_sentence = []
    words = sentence.split()
    for word in words:
        encoded_sentence.extend(codes[word])
    return encoded_sentence


def decode_sentence(encoded_sentence, root):
    decoded_sentence = []
    current_node = root
    for bit in encoded_sentence:
        if bit == 0:
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.word is not None:
            decoded_sentence.append(current_node.word)
            current_node = root

    return ' '.join(decoded_sentence)
def huffman_coding(sentences):
    # Build Huffman Tree
    root = build_huffman_tree(sentences)
    if not root:
        return [], {}, "", []

    # Generate Huffman Codes
    codes, code_lengths_dict = generate_huffman_codes(root)

    # Encode each sentence
    encoded_sentences = []
    code_lengths_list = []
    for sentence in sentences:
        encoded_sentence, code_lengths = encode_sentence(sentence, codes)
        encoded_sentences.append(encoded_sentence)
        code_lengths_list.append(code_lengths)

    return encoded_sentences, codes, root, code_lengths_list

def huffman_decoding(encoded_sentence, root):
    return decode_sentence(encoded_sentence, root)



#%%
import pandas as pd
test_data = list(pd.read_csv('test_data.csv')['Sememe'])
encoded_sentences, codes, root,code_length_list = huffman_coding(test_data)
start_idx = 500
end_idx = 701
inputCode = []
huff_length = [0]
# #%%
# huff_length = scipy.io.loadmat('huff_length_500_700.mat')['huff_length']
# inputCode = scipy.io.loadmat('inputCodes500_700.mat')['inputCodes']
for j in range(start_idx,end_idx):
    for b in encoded_sentences[j]:
        inputCode.append(b)
    for bb in range(int(np.ceil(len(code_length_list[j])/2))):
        #print(bb)
        try:
            hufff = code_length_list[j][bb*2] + code_length_list[j][bb*2+1]
        except:
            hufff = code_length_list[j][bb*2]
        huff_length.append(hufff)
    huff_length.append(0)
inputCode = np.array(inputCode).reshape(1,-1)
huff_length  = np.array(huff_length).reshape(1,-1)
#%%
device ='cuda'
#%%
def channel_equalized_noise(noise,fading):
    real_in1 = noise[:,:,0]
    comp_in1 = noise[:,:,1]
    real_in2 = fading[:,:,0]
    comp_in2 = fading[:,:,1]
    real_out = np.multiply(real_in1,real_in2) + np.multiply(comp_in1,comp_in2)
    comp_out = -np.multiply(real_in1,comp_in2) + np.multiply(comp_in1,real_in2)
    num = np.stack((real_out,comp_out),axis=2)
    denom_s = np.multiply(real_in2,real_in2)+np.multiply(comp_in2,comp_in2)
    denom = np.stack((denom_s,denom_s),axis=2)
    return np.divide(num,denom)
def rayleigh_channel(x,snr):
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
    x_2  = tf.stack((x_real,x_imag),axis=2)
    print(x_2.shape)
    fading = np.random.randn(x_2.shape[0],x_2.shape[1],x_2.shape[2])*(1/np.sqrt(2))
    n_power = 1 /(10 ** (snr / 10.0));
    #n_power = x_power / (10 ** (snr / 10.0))
    n_std = np.sqrt(n_power/2)
    noise = np.random.randn(x_2.shape[0],x_2.shape[1],x_2.shape[2]) *n_std
    equalized = x_2 + channel_equalized_noise(noise,fading)
    eqq = tf.complex(equalized[:,:,0],equalized[:,:,1])
    return eqq
# #%%
# encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k                 = 11, # number of information bits (input)
#                         n                 = 33) # number of codeword bits (output)


# decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(encoder           = encoder,
#                         num_iter          = 100, # number of BP iterations
#                         return_infobits   = True)
#%%
encoder = sionna.fec.polar.Polar5GEncoder(k          = 13, # number of information bits (input)
                          n          = 39) # number of codeword bits (output)


decoder = sionna.fec.polar.Polar5GDecoder(enc_polar    = encoder, # connect the Polar decoder to the encoder
                          dec_type   = "SCL", # can be also "SC" or "BP"
                          list_size  = 32)
#%%
polar_enc_dec = {}
for k in range(13,40):
    polar_enc = sionna.fec.polar.Polar5GEncoder(k          = k, # number of information bits (input)
                              n          = int(3*k)) # number of codeword bits (output)
    polar_dec = sionna.fec.polar.Polar5GDecoder(enc_polar    = polar_enc, # connect the Polar decoder to the encoder
                              dec_type   = "SCL", # can be also "SC" or "BP"
                              list_size  = 32)
    polar_enc_dec[k] = polar_enc,polar_dec
#%% AWGN
import time
modulator = sionna.mapping.Mapper(constellation_type = 'qam',num_bits_per_symbol=2)
demodulator = sionna.mapping.Demapper(demapping_method='maxlog',constellation_type = 'qam',num_bits_per_symbol=2)
awgn_channel = sionna.channel.AWGN()
decoded_all_awgn = []
for snr in range(-3,4):
    print('awgn ',snr)
    no = 1/(10**(snr/10))
    decoded_sent_list = []
    last_bit = 0
    concat_bit = tf.cast(np.array([0]).reshape(1,1),tf.float32)
    for i in range(huff_length.shape[1]):
        if huff_length[0,i] == 0:
            if i > 0:
                
                #print(dec_seq)
                decoded_sent_list.append(tf.concat(dec_seq,1))
                print(i)
            dec_seq = []
        else:
            c = inputCode[0,last_bit:last_bit+huff_length[0,i]]
            
            if c.shape[0] < 13:
                encoder,decoder = polar_enc_dec[13]
                need_zeros = 13-c.shape[0]
                c = np.append(c,np.zeros(need_zeros,))
                c = c.reshape(1,c.shape[0])
                c = tf.cast(c,tf.float32)
                encoded = encoder(c)
                encoded = tf.concat([encoded,concat_bit],1)
                modulated = modulator(encoded)
                y = awgn_channel((modulated,no))
                demodulated = demodulator((y,no))
                channel_decoded = decoder(demodulated[:,:-1])
                #print(channel_decoded)
                dec_seq.append(tf.cast(channel_decoded[:,:13-need_zeros],dtype='uint64'))
                last_bit += huff_length[0,i]
            else:
                encoder,decoder = polar_enc_dec[huff_length[0,i]]
                if huff_length[0,i] % 2 == 0:
                    start_0= time.time()
                    c = c.reshape(1,c.shape[0])
                    c = tf.cast(c,tf.float32)
                    #start_1 = time.time()
                    encoded = encoder(c)
                    #start_2 = time.time()
                    modulated = modulator(encoded)
                    #start_3 = time.time()
                    y = awgn_channel((modulated,no))
                    #start_4 = time.time()
                    demodulated = demodulator((y,no))
                    #start_5 = time.time()
                    channel_decoded = decoder(demodulated)
                    #start_6  =time.time()
                    # print('channel_deco',start_6-start_5)
                    # print(start_5-start_4)
                    # print(start_4-start_3)
                    # print(start_3-start_2)
                    # print(start_2-start_1)
                    # print(start_1-start_0)
                    dec_seq.append(tf.cast(channel_decoded,dtype=tf.uint64))
                    last_bit += huff_length[0,i]
                else:
                    c = c.reshape(1,c.shape[0])
                    c = tf.cast(c,tf.float32)
                    encoded = encoder(c)
                    encoded = tf.concat([encoded,concat_bit],1)
                    modulated = modulator(encoded)
                    y = awgn_channel((modulated,no))
                    demodulated = demodulator((y,no))
                    channel_decoded = decoder(demodulated[:,:-1])
                    #print(channel_decoded)
                    dec_seq.append(tf.cast(channel_decoded,dtype=tf.uint64))
                    last_bit += huff_length[0,i]
    decoded_all_awgn.append(decoded_sent_list)
decoded_all_rayleigh = []
for snr in range(0,13,2):
    print('rayleigh ', snr)
    no = 1/(10**(snr/10))
    decoded_sent_list = []
    last_bit = 0
    concat_bit = tf.cast(np.array([0]).reshape(1,1),tf.float32)
    for i in range(huff_length.shape[1]):
        if huff_length[0,i] == 0:
            if i > 0:
                decoded_sent_list.append(tf.concat(dec_seq,1))
                print(i)
            dec_seq = []
        else:
            c = inputCode[0,last_bit:last_bit+huff_length[0,i]]
            if c.shape[0] < 13:
                encoder,decoder = polar_enc_dec[13]
                need_zeros = 13-c.shape[0]
                c = np.append(c,np.zeros(need_zeros,))
                c = c.reshape(1,c.shape[0])
                c = tf.cast(c,tf.float32)
                encoded = encoder(c)
                encoded = tf.concat([encoded,concat_bit],1)
                modulated = modulator(encoded)
                #y = awgn_channel((modulated,no))
                y = rayleigh_channel(modulated,snr)
                demodulated = demodulator((y,no))
                channel_decoded = decoder(demodulated[:,:-1])
                #print(channel_decoded)
                dec_seq.append(channel_decoded[:,:13-need_zeros])
                last_bit += huff_length[0,i]
            else:
                encoder,decoder = polar_enc_dec[huff_length[0,i]]
                if huff_length[0,i] % 2 == 0:
                    start_0= time.time()
                    c = c.reshape(1,c.shape[0])
                    c = tf.cast(c,tf.float32)
                    #start_1 = time.time()
                    encoded = encoder(c)
                    #start_2 = time.time()
                    modulated = modulator(encoded)
                    #start_3 = time.time()
                    #y = awgn_channel((modulated,no))
                    y = rayleigh_channel(modulated,snr)
                    #start_4 = time.time()
                    demodulated = demodulator((y,no))
                    #start_5 = time.time()
                    channel_decoded = decoder(demodulated)
                    #start_6  =time.time()
                    # print('channel_deco',start_6-start_5)
                    # print(start_5-start_4)
                    # print(start_4-start_3)
                    # print(start_3-start_2)
                    # print(start_2-start_1)
                    # print(start_1-start_0)
                    dec_seq.append(channel_decoded)
                    last_bit += huff_length[0,i]
                else:
                    c = c.reshape(1,c.shape[0])
                    c = tf.cast(c,tf.float32)
                    encoded = encoder(c)
                    encoded = tf.concat([encoded,concat_bit],1)
                    modulated = modulator(encoded)
                    #y = awgn_channel((modulated,no))
                    y = rayleigh_channel(modulated,snr)
                    demodulated = demodulator((y,no))
                    channel_decoded = decoder(demodulated[:,:-1])
                    #print(channel_decoded)
                    dec_seq.append(channel_decoded)
                    last_bit += huff_length[0,i]
    decoded_all_rayleigh.append(decoded_sent_list)
#%%
huff_decoded_sentences_awgn = []
huff_decoded_sentences_rayleigh = []

for snr in range(7):
    #for i in range(2):
        #print(i)
    decoded_sentence = [huffman_decoding(encoded_sentence.numpy().tolist()[0], root) for encoded_sentence in decoded_all_awgn[snr]]
    huff_decoded_sentences_awgn.append(decoded_sentence)
for snr in range(7):
    #for i in range(200):
        #print(i)
    decoded_sentence = [huffman_decoding(encoded_sentence.numpy().tolist()[0], root) for encoded_sentence in decoded_all_rayleigh[snr]]
    huff_decoded_sentences_rayleigh.append(decoded_sentence)
#%%
import pickle
with open('decoded_sentences_awgn.pickle', 'wb') as handle:
    pickle.dump(huff_decoded_sentences_awgn, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('decoded_sentences_rayleigh.pickle', 'wb') as handle:
    pickle.dump(huff_decoded_sentences_rayleigh, handle, protocol=pickle.HIGHEST_PROTOCOL)
