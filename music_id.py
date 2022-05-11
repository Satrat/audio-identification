'''
Music Informatics 7006P
Coursework 2: Audio Identification

Sara Adkins
210077786
'''
import numpy as np
import os
import librosa
import librosa.display
from skimage.feature import peak_local_max
from scipy import ndimage, misc
import pickle

### hyperparameters ###
WINDOW_SIZE = 1024
HOP_SIZE = 256
SR = 22050
NUM_RANK = 3
NEIGH_WIDTH_F = 30
NEIGH_WIDTH_T = 20
TAR_WIDTH_F = 50
TAR_WIDTH_T = 70

### Helper functions for calculating transforms ###
def getStft(data, sr, window, hop):
    return np.abs(librosa.stft(data,n_fft=window,window='hann',win_length=window,hop_length=hop))
 
def getMel(data,sr, window, hop, n_mels=128,fmax=8000):
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels,fmax=fmax, win_length=window,hop_length=hop)
    return S

def getCQ(data, sr):
    C = np.abs(librosa.cqt(data, sr=sr))
    return C

def getConstellationBool(spectrogram, width_freq, width_time):
    '''
    Calculate spectral peaks using a maximum filter

    Input
        spectrogram: 2d array of time-frequency information
        width_freq: size of maximum filter in frequency axis
        width_time: size of maximum filter in time axis
    Output
        peaks: 2d boolean array, True where peaks are located
    '''
    maxed_filter = ndimage.maximum_filter(spectrogram, size=(width_freq,width_time))
    peaks = (maxed_filter == spectrogram) #find locations where spectrogram matches maximum
    return peaks #same size as spetrogram

def dbToPairIdx(const_bool):
    '''
    Computes the pairwise inverted index for a constellation map

    Input
        const_bool: 2d boolean array with peak locations
    Output
        hashes: dictionary mapping (f_anchor, f_target, time_diff) to anchor times
    '''
    coordinates = np.where(const_bool)
    tf_pts = zip(coordinates[1], coordinates[0]) #iterator of (time, freq) peak coordinates
    sorted_pts = sorted(tf_pts) # sort based on time then frequency
    hashes = {}
    for t,f in sorted_pts: #adding 1 to time so anchor point won't be a target
        target_zone = const_bool[f:(f+TAR_WIDTH_F),(t + 1):(t+TAR_WIDTH_T + 1)] 
        neighbors = np.where(target_zone)
        neighbors = zip(neighbors[1] + t + 1, neighbors[0] + f) #get coordinates of targets
        for neigh_t, neigh_f in neighbors: # add all targets to hash table
            hash_val = (f, neigh_f, (neigh_t - t))
            if hash_val in hashes:
                hashes[hash_val].append(t)
            else:
                hashes[hash_val] = [t]
    return hashes

def qToPairs(const_q):
    '''
    Computes the pairwise list of peaks for a constellation map

    Input
        const_q: 2d boolean array with peak locations
    Output
        hashes: list of (anchor_time, (f_anchor, f_target, time_diff)) tuples
    '''
    coordinates = np.where(const_q)
    tf_pts = zip(coordinates[1], coordinates[0])
    sorted_pts = sorted(tf_pts)
    queries = []
    for t,f in sorted_pts:
        target_zone = const_q[f:(f+TAR_WIDTH_F),(t + 1):(t+TAR_WIDTH_T + 1)]
        neighbors = np.where(target_zone)
        neighbors = zip(neighbors[1] + t + 1, neighbors[0] + f)
        for neigh_t, neigh_f in neighbors:
            queries.append((t,(f, neigh_f, (neigh_t - t)))) # store anchor time and hash value
    return queries

def matchConstellationsPairwise(d_inv, queries, d_len, q_len):
    '''
    Calculate the match between a database entry and query by shifting. Returns the most promising
    shift result

    Input
        d_inv: pairwise inverted index dictionary for database item
        queries: list of times and hash values for query item
        d_len: length in frames of database spectrogram
        q_len: length in frames of query spectrogram
    Output
        max_shift: shift value producing the best match
        max_val: top matching function value
        matching: list of all shifted matches, used for plotting
    '''

    #all possible overlaps between query and database
    min_shift = -1 * q_len
    max_shift = d_len + q_len
    shift_rng = max_shift - min_shift

    num_Q = len(queries)
    indicators = np.zeros((num_Q, shift_rng)) #store indicator fn output for each query peak and shift
    for i,(t,hash_val) in enumerate(queries):
        inv_idx = []
        if hash_val in d_inv:
            inv_idx = d_inv[hash_val].copy() #get all timestamps that match the hash
        inv_idx = inv_idx - t + q_len # calculate corresponding shift value
        for idx in inv_idx:
            indicators[i][idx] = 1 #mark a hit for query i with shift idx
            
    #find the best match
    matching = np.sum(indicators, axis=0) 
    max_idx = np.argmax(matching)
    max_val = matching[max_idx]
    max_shift = max_idx - q_len
    
    return max_shift, max_val, matching

def fingerprintFromFilePairwise(filename, stft_type='stft'):
    '''
    Calculate spectral peaks from a wav file

    Input:
        filename: wav file to read
        stft_type: stft, mel or cq
    Output
        const: 2d boolean array of spectrogram peaks
        frames: number of frames in spectrogram
    '''
    y, sr = librosa.load(filename, sr=SR)
    if(stft_type == 'stft'):
        spec = getStft(y, SR, WINDOW_SIZE, HOP_SIZE)
    elif(stft_type == 'mel'):
        spec = getMel(y, SR, WINDOW_SIZE, HOP_SIZE)
    elif(stft_type == 'cq'):
        spec = getCQ(y, SR)
    const = getConstellationBool(np.log(spec + 1e-7), NEIGH_WIDTH_F,NEIGH_WIDTH_T)
    
    return const, spec.shape[1]

def fingerprintBuilder(db_folder, fp_path, stft_type='stft'):
    '''
    Construct a fingerprint in the form of pairwise inverted index for a folder of wavs

    Input
        db_folder: folder containing wav files of all database items
        fp_path: pickle file path to write fingerprints to
        stft_type: stft, mel or cq
    Output
        db_fingerprints: dictionary mapping each file name to its pairwise inverted index and frame length
    '''
    db_fingerprints = {}
    for file in os.listdir(db_folder):
        full_path = os.path.join(db_folder,file)
        constellation, frames = fingerprintFromFilePairwise(full_path, stft_type=stft_type) #get peaks
        inv_idx = dbToPairIdx(constellation) #calculate fingerprint using hashing
        db_fingerprints[file] = inv_idx, frames
    
    with open(fp_path, 'wb') as out_f:
        pickle.dump(db_fingerprints, out_f)
    return db_fingerprints

def audioIdentification(q_folder, fp_path, output_path, stft_type='stft'):
    '''
    Find the top 3 database matches for all queries in q_folder. Write results to output_path
    Input
        q_folder: folder containing wav files of queries
        fp_path: path to pickle file containing dictionary of pairwise inverted indices for each database item
        output_path: text file to write results to
        stft_type: stft, mel or cq
    Output
        None
    '''
    f = open(fp_path, "rb")
    fingerprints = pickle.load(f)
    f.close()

    f = open(output_path, "w")
    for file in os.listdir(q_folder):
        full_path = os.path.join(q_folder, file)
        q_fp, q_len = fingerprintFromFilePairwise(full_path,stft_type=stft_type)
        queries = qToPairs(q_fp) #get query hashes
        matches = []
        for name, (inv_idx, d_len) in fingerprints.items(): #calculate highest match for each db item
            shift, max_val, _ = matchConstellationsPairwise(inv_idx, queries, d_len, q_len)
            matches.append((name, shift, max_val))
        
        matches.sort(key=lambda tup: tup[2], reverse=True) #sort by maximum match score
        top_matches = [m[0] for m in matches[0:NUM_RANK]] #get filenames of top 3 matches
        f.write("{}\t{}\t{}\t{}\n".format(file, top_matches[0], top_matches[1], top_matches[2]))
    f.close()

def calcPrecisionRecall(query_string):
    '''
    Calculate precision and recall for a single query

    Input
        query_string: tab separated string containing query name and top 3 matches
    Output
        precision: list of precision calculations for rank 1-3
        recall: list of recall calculations for rank 1-3
    '''
    split_string = query_string.split('\t')
    query_f = split_string[0]
    matches = [os.path.splitext(s)[0] for s in split_string[1::]]
    
    precision = [0.0] * NUM_RANK
    recall = [0.0] * NUM_RANK
    total_hits = 0
    for i in range(NUM_RANK): #calculate cummulative precision/recall
        if matches[i] in query_f:
            total_hits += 1
        precision[i] = total_hits * 1.0 / (i + 1)
        recall[i] = total_hits #only one match for query, no need to divide by I_Q
        
    return precision, recall

def calcAvgPrecision(output_path):
    '''
    Calculate the average precision across all queries

    Input
        output_path: tab separated text file containing top 3 matches for each query
    Output
        avg_precision: average precision value across all queries

    '''
    f = open(output_path, "r")
    output_data = f.readlines()
    f.close()
    
    total_queries = len(output_data)
    sum_precision = 0.0
    for line in output_data:
        p, r = calcPrecisionRecall(line)

        #1.0 if answer in slot 1, 0.5 for slot 2, 0.333 for slot 3
        sum_precision += np.max(p)
        
    return sum_precision / total_queries