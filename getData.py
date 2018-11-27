import xlrd
import random
import operator
import numpy as np
from scipy.stats import bernoulli

"""
methods to get real data or generate simulation data
"""

names = ['GDP', 'CPI', 'CCPI', 'RS', 'HS', 'NHPI', 'CA', 'D.UE', 'D.Brent', 'D.WTI']


# get Canada economy data
def get_economy_data():
    workbook = xlrd.open_workbook("data/data.xlsx")
    table = workbook.sheets()[0]
    result = {}

    for i in range(1, 11):
        cols = table.col_values(i)
        result[cols[0]] = cols[1:113]
        print cols[0]
    return result


# generate Bernoulli distribution with changing probability of success
def get_b_data(p, p_value, length, noise):
    cause = []
    data_bern = bernoulli.rvs(size=p, p=p_value)
    cause.extend(data_bern)
    for i in range(0, length - p):
        num = str(cause).count("1")
        p_tmp = float(num) / len(cause)
        if p_tmp > 0.9:
            p_tmp = 0.9
        elif p_tmp < 0.1:
            p_tmp = 0.1
        new_element = bernoulli.rvs(size=1, p=p_tmp)[0]
        cause.append(new_element)
    effect = forward_shift_zero_one_data(cause, random.randint(1, 5))
    effect = add_noise(effect, noise)
    print len(cause)
    print len(effect)
    return cause, effect


# turn economic data into 01 data with growth and reduction as indicators
def change_economy_data_to_zero_one_data():
    data_dic = get_economy_data()
    result = {}
    for name in names:
        if name in ['GDP', 'CPI', 'CCPI', 'RS', 'NHPI', 'D.UE']:
            zero_one_data = []
            data = data_dic[name]
            for element in data:
                if element > 0:
                    zero_one_data.append(1)
                    zero_one_data.append(0)
                elif element < 0:
                    zero_one_data.append(0)
                    zero_one_data.append(1)
                else:
                    zero_one_data.append(0)
                    zero_one_data.append(0)
            result[name] = zero_one_data
        elif name in ['HS', 'CA']:
            zero_one_data = [0, 0]
            data = data_dic[name]
            for i in range(1, len(data)):
                if data[i] - data[i - 1] > 0 and abs(data[i] - data[i - 1]) / abs(data[i - 1]) >= 0.05:
                    zero_one_data.append(1)
                    zero_one_data.append(0)
                elif data[i] - data[i - 1] < 0 and abs(data[i] - data[i - 1]) / abs(data[i - 1]) >= 0.05:
                    zero_one_data.append(0)
                    zero_one_data.append(1)
                else:
                    zero_one_data.append(0)
                    zero_one_data.append(0)
            result[name] = zero_one_data
        elif name in ['D.Brent', 'D.WTI']:
            zero_one_data = []
            data = data_dic[name]
            for element in data:
                if element >= 0.05:
                    zero_one_data.append(1)
                    zero_one_data.append(0)
                elif element <= -0.05:
                    zero_one_data.append(0)
                    zero_one_data.append(1)
                else:
                    zero_one_data.append(0)
                    zero_one_data.append(0)
            result[name] = zero_one_data
    return result


# Bernoulli distribution with a probability of success of 0.5
def generate_random_zero_one_data(size):
    cause = []
    for i in range(0, size):
        cause.append(random.randint(0, 1))
    return cause


# move the 01 sequence forward by shift bits
def forward_shift_zero_one_data(seq, shift):
    lseq = len(seq)
    sseq = [None] * lseq
    for i in xrange(lseq):
        if i >= shift:
            sseq[i] = seq[i - shift]
        else:
            sseq[i] = random.choice([0, 1])
    return sseq


# move the coninue sequence forward by shift bits
def forward_shift_continue_data(seq, shift):
    lseq = len(seq)
    sseq = [None] * lseq
    for i in xrange(lseq):
        if i >= shift:
            sseq[i] = seq[i - shift]
        else:
            sseq[i] = np.random.normal(0, 1, 1)[0]
    return sseq


# add noise to 01 sequence
def add_noise(seq, noise):
    for i in range(0, len(seq)):
        if random.random() < noise:
            if seq[i] == 1:
                seq[i] = 0
            elif seq[i] == 0:
                seq[i] = 1
    return seq


# forward shift and add noise to generate effect
def generate_effect_of_one_zero_data(seq, shift, noise):
    effect = forward_shift_zero_one_data(seq, shift)
    effect = add_noise(effect, noise)
    return effect


# generate data way in former essay
def gen_cause(size, p):
    cause = [random.randint(0, 1) for i in xrange(p)]
    for i in xrange(p, size):
        i1 = random.randint(i - p, i - 2)
        i2 = random.randint(i1 + 1, i - 1)
        c = reduce(operator.xor, cause[i1:i2])
        cause.append(c)
    return cause


# generate continue cause and effect
def generate_continue_data(length, shift):
    cause = []
    main = np.random.normal(0, 1, length)
    noise = np.random.normal(0, 0.1, length)
    for i in range(0, length):
        if i == 0:
            cause.append(main[i])
        else:
            cause.append(cause[i - 1] + main[i])
    effect = forward_shift_continue_data(cause, shift)
    for j in range(0, length):
        effect[j] = effect[j] + noise[j]
    return cause, effect


def change_continue_to_one_zero(seq):
    result = []
    for i in range(1, len(seq)):
        if seq[i] - seq[i - 1] >= 0:
            result.append(1)
        else:
            result.append(0)
    return result


# a random walk based on a normal distribution
def ge_normal_data(p, length):
    cause = []
    main = np.random.normal(0, 1, p)
    main_final = []
    main_final.extend(main)
    noise = np.random.normal(0, 0.1, p)
    noise_final = []
    noise_final.extend(noise)
    for i in range(0, length - p):
        new_element = np.random.normal(0, np.std(main_final), 1)[0]
        main_final.append(new_element)
    for j in range(0, length - p):
        new_element = np.random.normal(0, np.std(noise_final), 1)[0]
        noise_final.append(new_element)
    print len(main_final)
    print len(noise_final)
    for i in range(0, length):
        if i == 0:
            cause.append(main_final[i])
        else:
            cause.append(cause[i - 1] + main_final[i])
    effect = forward_shift_continue_data(cause, p)
    for j in range(0, length):
        effect[j] = effect[j] + noise_final[j]
    return cause, effect
