import numpy as np
import pandas as pd

import random
from collections import Counter
import copy

import warnings
warnings.filterwarnings("ignore")

eng = pd.read_csv('/data/alphabet.csv', sep=',',header=None)
P = pd.read_csv('/data/letter_probabilities.csv', sep=',',header=None)
M = pd.read_csv('/data/letter_transition_matrix.csv', sep=',',header=None)
# M dims: idx 1 is input char, idx 2 is output char
eng_dict = {eng[i][0]: i for i in range(28)}

P = np.log(P)
M = np.log(M)

MOMENTUM_K = 5
PERTURB = 1.5e-3
WINDOW_SIZE = 500
NATS_THRESHOLD = -2.375
BURN_IN_TIME = 1500


def decipher(ciphertext, f):
    inv_f = {eng[f[i]][0]: eng[i][0] for i in range(28)}
    return ''.join([inv_f[ch] for ch in ciphertext])

def log_likelihood(plaintext):
    return P[eng_dict[plaintext[0]]][0] + \
        sum([M[eng_dict[plaintext[i]]][eng_dict[plaintext[i+1]]] for i in range(len(plaintext)-1)])

def fastest_log_likelihood(memo, bigram_dict, f):
    if f in memo:
        return memo[f]
    else:
        ll = 0
        inv_f = {f[i]: i for i in range(28)}
        for j in range(28):
            for k in range(28):
                bigram_count = bigram_dict[j][k]
                if bigram_count != 0:
                    ll += bigram_count*M[inv_f[j]][inv_f[k]]
        memo[f] = ll
        return memo[f]

# unused: SGD analogue
def stochastic_log_likelihood(plaintext):
    rand_idx = random.randint(0, len(plaintext)*7//8-1)
    rand_subtext = plaintext[rand_idx: rand_idx + len(plaintext)//8]
    return log_likelihood(rand_subtext)

def Metropolis_Hastings(ciphertext, for_bp = False, checkpoint = None):
    bigram_dict = {j: {k: 0 for k in range(28)} for j in range(28)}
    for i in range(len(ciphertext)-1):
        bigram_dict[eng_dict[ciphertext[i]]][eng_dict[ciphertext[i+1]]] += 1
    fn = checkpoint if checkpoint is not None else tuple(np.random.permutation(28))
    accepted = [0, 0, 0, 0, 1] # MOMENTUM_K = 5
    epoch = 0
    old_ll = log_likelihood(decipher(ciphertext, fn))
    memo = {}
    memo[fn] = old_ll
    freqs = {}
    freqs[fn] = 1
    while True:
        swap = random.sample(range(28), 2)
        fprime = list(fn)
        fprime[swap[0]], fprime[swap[1]] = fprime[swap[1]], fprime[swap[0]]
        fprime = tuple(fprime)
        prop_ll = fastest_log_likelihood(memo, bigram_dict, fprime)
        acceptance_factor = min(0, prop_ll-old_ll)
        ber_flip = random.random()
        # add momentum analogue
        momentum_factor = 1 + 0.4*sum(accepted)
        acceptance_factor *= momentum_factor
        # add bypass perturbation
        if np.log(ber_flip) < acceptance_factor or random.random() < PERTURB:
            fn = fprime
            accepted[epoch % 5] = 1
            old_ll = prop_ll
            freqs[fprime] = freqs.get(fprime, 0) + 1
        else:
            accepted[epoch % MOMENTUM_K] = 0
            freqs[fn] = freqs.get(fn, 0) + 1
        epoch += 1
        # discard before prescribed burn-in time
        if epoch == BURN_IN_TIME:
            freqs = {}
        # dynamical termination: log-likelihood vs. bigram entropy test
        if epoch > BURN_IN_TIME and epoch % 500 == 0:
            map_cipher = max(memo, key=memo.get)
            cur_ll = memo[map_cipher]
            if cur_ll/len(ciphertext) > NATS_THRESHOLD:
                #print('HIT!')
                #print('epoch ' + str(epoch))
                #print('log likelihood ' +str(cur_ll))
                return decipher(ciphertext, map_cipher), map_cipher, cur_ll/len(ciphertext)
        # terminate under 120 secs, or if for breakpoint search, terminate early
        if epoch > 80000 or (for_bp and epoch > 20000):
            map_cipher = max(memo, key=memo.get)
            cur_ll = memo[map_cipher]
            return decipher(ciphertext, map_cipher), map_cipher, cur_ll/len(ciphertext)
        # early terminate for breakpoint search


def Binary_MCMC(ciphertext):
    lp, rp = 0, len(ciphertext)-1
    bp = (lp+rp)//2
    l_cipher, r_cipher = None, None
    final_l, final_r = None, None
    BP_THRESH = NATS_THRESHOLD - 0.05
    while final_l is None and final_r is None:
        _, l_cipher, l_normll = Metropolis_Hastings(ciphertext[:bp], for_bp=True, checkpoint=l_cipher)
        if l_normll > BP_THRESH:
            final_l = l_cipher
            lp = bp
            #print("left checkpoint " + str(l_normll))
        _, r_cipher, r_normll = Metropolis_Hastings(ciphertext[bp:], for_bp=True, checkpoint=r_cipher)
        if r_normll > BP_THRESH:
            final_r = r_cipher
            rp = bp
            #print("right checkpoint " + str(r_normll))
        bp = (lp+rp)//2
    while lp-rp > 8:
        if final_l:
            l_normll = log_likelihood(decipher(ciphertext[:bp], final_l))/len(ciphertext[:bp])
            if l_normll > BP_THRESH:
                lp = bp
            else:
                rp = bp
            bp = (lp+rp)//2
        if final_r:
            r_normll = log_likelihood(decipher(ciphertext[bp:], final_r))/len(ciphertext[bp:])
            if r_normll > BP_THRESH:
                rp = bp
            else:
                lp = bp
            bp = (lp+rp)//2
    if final_l:
        r_ans, final_r, _ = Metropolis_Hastings(ciphertext[bp:], for_bp=True, checkpoint=r_cipher)
        l_ans = decipher(ciphertext[:bp], final_l)
    elif final_r:
        l_ans, final_l, _ = Metropolis_Hastings(ciphertext[:bp], for_bp=True, checkpoint=l_cipher)
        r_ans = decipher(ciphertext[bp:], final_r)
    return l_ans + r_ans


def decode(ciphertext: str, has_breakpoint: bool) -> str:
    if has_breakpoint:
        return Binary_MCMC(ciphertext)
    plaintext, _, _ = Metropolis_Hastings(ciphertext)
    return plaintext

if __name__ == "__main__":
    ciphertext = None
    plaintext = None
    with open('/data/sample/short_ciphertext_breakpoint.txt') as f:
        lines = f.readlines()
        ciphertext = lines[0]
    with open('/data/sample/short_plaintext.txt') as f:
        lines = f.readlines()
        plaintext = lines[0]
    out = decode(ciphertext, True)
    print(sum(int(i == j) for i, j in zip(out, plaintext)))

'''
if __name__ == "__main__":
    ciphertext = None
    plaintext = None
    with open('../data/sample/short_ciphertext.txt') as f:
        lines = f.readlines()
        ciphertext = lines[0]
    with open('../data/sample/short_plaintext.txt') as f:
        lines = f.readlines()
        plaintext = lines[0]
    truecipher = ciphertext
    split1000 = ciphertext[0:len(ciphertext)//2]
    ciphertext = ciphertext[len(ciphertext)//2:]
    split500 = ciphertext[0:len(ciphertext)//2]
    ciphertext = ciphertext[len(ciphertext)//2:]
    split250 = ciphertext[0:len(ciphertext)//2]
    ciphertext = ciphertext[len(ciphertext)//2:]
    split125 = ciphertext[0:len(ciphertext)//2]
    ciphertext = ciphertext[len(ciphertext)//2:]

    trueplain = plaintext
    plain1000 = plaintext[0:len(plaintext)//2]
    plaintext = plaintext[len(plaintext)//2:]
    plain500 = plaintext[0:len(plaintext)//2]
    plaintext = plaintext[len(plaintext)//2:]
    plain250 = plaintext[0:len(plaintext)//2]
    plaintext = plaintext[len(plaintext)//2:]
    plain125 = plaintext[0:len(plaintext)//2]
    plaintext = plaintext[len(plaintext)//2:]

    for split, plain in zip([split1000, split500, split250, split125], [plain1000, plain500, plain250, plain125]):
        random_walk, _ = Metropolis_Hastings(split)
        map_cipher = Counter(random_walk).most_common(1)[0][0]
        output_plaintext = decipher(split, map_cipher)
        print(log_likelihood(output_plaintext))
        print(sum(int(i == j) for i, j in zip(output_plaintext, plain)))
        print('\n\n')
'''
'''
diff_ll = 0
diff_ll += sum([(new_bigram_dict[plain_swap[k]][eng[j][0]]-old_bigram_dict[plain_swap[k]][eng[j][0]])\
            *M[eng_dict[plain_swap[k]]][j] for j in range(28) for k in (0, 1)])
diff_ll += sum([(new_bigram_dict[eng[j][0]][plain_swap[k]]-old_bigram_dict[eng[j][0]][plain_swap[k]])\
            *M[j][eng_dict[plain_swap[k]]] for j in range(28) for k in (0, 1)])
diff_ll -= sum([(new_bigram_dict[plain_swap[j]][plain_swap[k]]-old_bigram_dict[plain_swap[j]][plain_swap[k]])\
            *M[eng_dict[plain_swap[j]]][eng_dict[plain_swap[k]]] for j in (0, 1) for k in (0, 1)])
new_ll = old_ll + diff_ll
'''
