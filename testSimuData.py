# -*- coding: utf8 -*-
import numpy as np
import random
from snml import bernoulli, cbernoulli
from statsmodels.tsa.stattools import grangercausalitytests
from getData import generate_effect_of_one_zero_data, gen_cause, generate_continue_data, change_continue_to_one_zero, \
    get_b_data, ge_normal_data
import math


# test the performance of the model under noisy conditions
def test_simulation_data(sample_num, length, noise):
    p = 5
    correct_cute, wrong_cute, correct_cute_undecided, wrong_cute_undecided = 0, 0, 0, 0
    bi_granger, correct_ganger, wrong_ganger, no_granger = 0, 0, 0, 0
    ncorrect_cute, nwrong_cute, nindec_cute = 0, 0, 0
    ncorrect_granger, nwrong_granger, nindec_granger = 0, 0, 0
    for i in range(0, sample_num):
        shift = random.randint(1, p)
        p_value = random.uniform(0.1, 0.9)
        # cause, effect = get_b_data(p,p_value,length,noise)
        cause = gen_cause(length, p)
        # cause = generate_random_zero_one_data(length)
        # p = random.choice((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
        effect = generate_effect_of_one_zero_data(cause, shift, noise)
        # effect = gen_cause(length, p)
        delta_ce = bernoulli(effect) - cbernoulli(effect, cause)
        delta_ec = bernoulli(cause) - cbernoulli(cause, effect)
        if delta_ce - delta_ec >= -math.log(0.05, 2):
            correct_cute += 1
        elif delta_ec - delta_ce >= -math.log(0.05, 2):
            wrong_cute += 1
        elif delta_ce - delta_ec > 0 and delta_ce - delta_ec <= -math.log(0.05, 2):
            correct_cute_undecided += 1
        else:
            wrong_cute_undecided += 1
        ncorrect_cute += int(delta_ce > delta_ec)
        nwrong_cute += int(delta_ce < delta_ec)
        nindec_cute += int(delta_ce == delta_ec)
        flag1 = False
        x = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        flag2 = False
        y = grangercausalitytests([[cause[i], effect[i]] for i in range(0, len(cause))], 5)
        for key in x:
            if x[key][0]["params_ftest"][1] < 0.05:
                flag1 = True
        for key in y:
            if y[key][0]["params_ftest"][1] < 0.05:
                flag2 = True
        if flag1:
            ncorrect_granger += 1
        else:
            nwrong_granger += 1
        if flag1 and flag2:
            bi_granger += 1
        elif flag1 and not flag2:
            correct_ganger += 1
        elif not flag1 and flag2:
            wrong_ganger += 1
        else:
            no_granger += 1
    return correct_cute, wrong_cute, correct_cute_undecided, wrong_cute_undecided, bi_granger, correct_ganger, wrong_ganger, no_granger


# test the performance of the model under noisy conditions but there is no causality
def test_simulation_data_no_causal(sample_num, length, noise):
    p = 5
    ncorrect_cute, nwrong_cute, nindec_cute = 0, 0, 0
    ncorrect_granger, nwrong_granger, nindec_granger = 0, 0, 0
    for i in range(0, sample_num):
        cause = gen_cause(length, p)
        # cause = generate_random_zero_one_data(length)
        # p = random.choice((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
        effect = gen_cause(length, p)
        delta_ce = bernoulli(effect) - cbernoulli(effect, cause)
        delta_ec = bernoulli(cause) - cbernoulli(cause, effect)
        if delta_ce - delta_ec > 1.3:
            ncorrect_cute += 1
        elif delta_ce - delta_ec < -1.3:
            nwrong_cute += 1
        else:
            nindec_cute += 1
        flag1 = False
        x = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        for key in x:
            if x[key][0]["params_ftest"][1] < 0.05:
                flag1 = True

        if flag1:
            ncorrect_granger += 1
        else:
            nwrong_granger += 1
    print "%.2fN ✓=%d ✘=%d ~=%d \t %.2f%%" % (
        noise, ncorrect_cute, nwrong_cute, nindec_cute, ncorrect_cute / float(sample_num))
    print "%.2fN ✓=%d ✘=%d ~=%d \t %.2f%%" % (
        noise, ncorrect_granger, nwrong_granger, nindec_granger, ncorrect_granger / float(sample_num))
    print
    return ncorrect_cute / float(sample_num), ncorrect_granger / float(sample_num)


# test data with different noise
def add_noise_test():
    noises = [0.0, 0.1, 0.2, 0.3, 0.4]
    correct_cutes = []
    wrong_cutes = []
    correct_cute_undecideds = []
    wrong_cute_undecideds = []
    bi_grangers = []
    correct_gangers = []
    wrong_gangers = []
    no_grangers = []
    for noise in noises:
        correct_cute, wrong_cute, correct_cute_undecided, wrong_cute_undecided, bi_granger, correct_ganger, wrong_ganger, no_granger = test_simulation_data(
            1000, 400, noise)
        correct_cutes.append(correct_cute)
        wrong_cutes.append(wrong_cute)
        correct_cute_undecideds.append(correct_cute_undecided)
        wrong_cute_undecideds.append(wrong_cute_undecided)
        bi_grangers.append(bi_granger)
        correct_gangers.append(correct_ganger)
        wrong_gangers.append(wrong_ganger)
        no_grangers.append(no_granger)
    print correct_cutes
    print wrong_cutes
    print correct_cute_undecideds
    print wrong_cute_undecideds
    print bi_grangers
    print correct_gangers
    print wrong_gangers
    print no_grangers


# check whether the continuous data and 01 data causality are consistent
def consistency_check():
    correct111 = 0
    correct100 = 0
    correct010 = 0
    correct001 = 0
    correct110 = 0
    correct101 = 0
    correct011 = 0
    correct000 = 0
    for i in range(0, 1000):
        p = random.randint(1, 5)
        cause, effect = generate_continue_data(200, p)
        effect, effect_test = generate_continue_data(200, p)
        cause2 = change_continue_to_one_zero(cause)
        effect2 = change_continue_to_one_zero(effect)
        flag1 = False
        x = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        for key in x:
            if x[key][0]["params_ftest"][1] < 0.05:
                flag1 = True
        flag2 = False
        y = grangercausalitytests([[effect2[i], cause2[i]] for i in range(0, len(cause2))], 5)
        for key in y:
            if y[key][0]["params_ftest"][1] < 0.05:
                flag2 = True
        flag3 = False
        delta_ce = bernoulli(effect2) - cbernoulli(effect2, cause2)
        delta_ec = bernoulli(cause2) - cbernoulli(cause2, effect2)
        if delta_ce > delta_ec and delta_ce - delta_ec > 1.3:
            flag3 = True
        if flag1 and flag2 and flag3:
            correct111 += 1
        elif flag1 and not flag2 and not flag3:
            correct100 += 1
        elif not flag1 and flag2 and not flag3:
            correct010 += 1
        elif not flag1 and not flag2 and flag3:
            correct001 += 1
        elif flag1 and flag2 and not flag3:
            correct110 += 1
        elif flag1 and not flag2 and flag3:
            correct101 += 1
        elif not flag1 and flag2 and flag3:
            correct011 += 1
        elif not flag1 and not flag2 and not flag3:
            correct000 += 1
    print correct111
    print correct100
    print correct010
    print correct001
    print correct110
    print correct101
    print correct011
    print correct000


# test continue data
def continue_test():
    counter = 0
    counter2 = 0
    total = 0
    for i in range(0, 1000):
        p = random.randint(1, 5)
        cause, test1 = generate_continue_data(200, p)
        effect, test2 = generate_continue_data(200, p)
        flag1 = False
        x = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        for key in x:
            if x[key][0]["params_ftest"][1] < 0.05:
                flag1 = True
        if flag1:
            continue
        total += 1
        cause2 = change_continue_to_one_zero(cause)
        effect2 = change_continue_to_one_zero(effect)
        flag2 = False
        y = grangercausalitytests([[effect2[i], cause2[i]] for i in range(0, len(cause2))], 5)
        for key in y:
            if y[key][0]["params_ftest"][1] < 0.01:
                flag2 = True
        flag3 = False
        delta_ce = bernoulli(effect2) - cbernoulli(effect2, cause2)
        delta_ec = bernoulli(cause2) - cbernoulli(cause2, effect2)
        if delta_ce > delta_ec and delta_ce - delta_ec > 2:
            flag3 = True
        if flag2:
            counter += 1
        if flag3:
            counter2 += 1
        print flag2
        print flag3
    print counter
    print counter2
    print total


# test causal continue data
def test_data():
    txtName = "causal_continue_noise_0.1_normal_prove.txt"
    f = file(txtName, "a+")
    counter11 = 0
    counter10 = 0
    counter01 = 0
    counter00 = 0
    counter11_01 = 0
    counter10_01 = 0
    counter01_01 = 0
    counter00_01 = 0
    counter_undecided = 0
    counter_true = 0
    counter_false = 0
    for i in range(0, 100):
        write_str = ""
        p = random.randint(1, 5)
        cause, effect = ge_normal_data(p, 200)
        for ii in range(0, len(cause)):
            write_str = write_str + " " + str(cause[ii])
        for jj in range(0, len(effect)):
            write_str = write_str + " " + str(effect[jj])
        print "cause:" + str(cause)
        print "effect:" + str(effect)
        # effect, test2 = ge_normal_data(p,200)
        print "Continuous data, Granger causality test"
        print "cause->effect"
        p_value_cause_to_effect1 = []
        flag1 = False
        ce1 = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        for key in ce1:
            p_value_cause_to_effect1.append(ce1[key][0]["params_ftest"][1])
            if ce1[key][0]["params_ftest"][1] < 0.05:
                flag1 = True
        print "effect->cause"
        p_value_effect_to_cause2 = []
        flag2 = False
        ce2 = grangercausalitytests([[cause[i], effect[i]] for i in range(0, len(cause))], 5)
        for key in ce2:
            p_value_effect_to_cause2.append(ce2[key][0]["params_ftest"][1])
            if ce2[key][0]["params_ftest"][1] < 0.05:
                flag2 = True
        if flag1 and flag2:
            print "Continuous data，Granger two-way cause and effect"
            write_str = write_str + " " + "Continuous data，Granger two-way cause and effect"
            counter11 += 1
        elif flag1 and not flag2:
            print "Continuous data，Granger correct cause and effect"
            write_str = write_str + " " + "Continuous data，Granger correct cause and effect"
            counter10 += 1
        elif not flag1 and flag2:
            print "Continuous data，Granger wrong cause and effect"
            write_str = write_str + " " + "Continuous data，Granger wrong cause and effect"
            counter01 += 1
        elif not flag1 and not flag2:
            print "Continuous data，Granger no cause and effect"
            write_str = write_str + " " + "Continuous data，Granger no cause and effect"
            counter00 += 1
        write_str = write_str + " " + str(min(p_value_cause_to_effect1)) + " " + str(min(p_value_effect_to_cause2))
        print
        cause2 = change_continue_to_one_zero(cause)
        effect2 = change_continue_to_one_zero(effect)
        print "01 data, Granger causality test"
        print "cause->effect"
        p_value_cause_to_effect3 = []
        flag3 = False
        ce3 = grangercausalitytests([[effect2[i], cause2[i]] for i in range(0, len(cause2))], 5)
        for key in ce3:
            p_value_cause_to_effect3.append(ce3[key][0]["params_ftest"][1])
            if ce3[key][0]["params_ftest"][1] < 0.05:
                flag3 = True
        print "effect->cause"
        p_value_effect_to_cause4 = []
        flag4 = False
        ce4 = grangercausalitytests([[cause2[i], effect2[i]] for i in range(0, len(cause2))], 5)
        for key in ce4:
            p_value_effect_to_cause4.append(ce4[key][0]["params_ftest"][1])
            if ce4[key][0]["params_ftest"][1] < 0.05:
                flag4 = True
        if flag3 and flag4:
            print "01 data，Granger two-way cause and effect"
            write_str = write_str + " " + "01 data，Granger two-way cause and effect"
            counter11_01 += 1
        elif flag3 and not flag4:
            print "01 data，Granger correct cause and effect"
            write_str = write_str + " " + "01 data，Granger correct cause and effect"
            counter10_01 += 1
        elif not flag3 and flag4:
            print "01 data，Granger wrong cause and effect"
            write_str = write_str + " " + "01 data，Granger wrong cause and effect"
            counter01_01 += 1
        elif not flag3 and not flag4:
            print "01 data，Granger no cause and effect"
            write_str = write_str + " " + "01 data，Granger no cause and effect"
            counter00_01 += 1
        write_str = write_str + " " + str(min(p_value_cause_to_effect3)) + " " + str(min(p_value_effect_to_cause4))
        print
        delta_ce = bernoulli(effect2) - cbernoulli(effect2, cause2)
        delta_ec = bernoulli(cause2) - cbernoulli(cause2, effect2)
        if delta_ce > delta_ec and delta_ce - delta_ec >= -math.log(0.05, 2):
            print "CUTE，correct cause and effect"
            write_str = write_str + " " + "CUTE，correct cause and effect"
            counter_true += 1
        elif delta_ec > delta_ce and delta_ec - delta_ce >= -math.log(0.05, 2):
            print "CUTE，wrong cause and effect"
            write_str = write_str + " " + "CUTE，wrong cause and effect"
            counter_false += 1
        else:
            print "CUTE，undecided"
            write_str = write_str + " " + "CUTE，undecided"
            counter_undecided += 1
        write_str = write_str + " " + str(pow(2, -abs(delta_ce - delta_ec)))
        f.write(write_str)
        f.write("\n")
        print
        print "*****************************cut line*****************************"
        print
    f.close()
    print "Continuous data, Granger causality test："
    print "two-way cause and effect:" + str(counter11)
    print "correct cause and effect:" + str(counter10)
    print "wrong cause and effect:" + str(counter01)
    print "no cause and effect" + str(counter00)
    print "-----------------"
    print "01 data，Granger causality test："
    print "two-way cause and effect:" + str(counter11_01)
    print "correct cause and effect:" + str(counter10_01)
    print "wrong cause and effect:" + str(counter01_01)
    print "no cause and effect" + str(counter00_01)
    print "-----------------"
    print "01 data，CUTE causality test："
    print "correct cause and effect:" + str(counter_true)
    print "wrong cause and effect:" + str(counter_false)
    print "no cause and effect:" + str(counter_undecided)


def test_random_zero_one_data():
    txtName = "causal_zero_one_normal_prove.txt"
    f = file(txtName, "a+")
    counter11_01 = 0
    counter10_01 = 0
    counter01_01 = 0
    counter00_01 = 0
    counter_true = 0
    counter_false = 0
    counter_undecided = 0
    for i in range(0, 100):
        write_str = ""
        p_value = random.uniform(0.1, 0.9)
        cause, effect = get_b_data(10, p_value, 200, 0)
        print "cause:" + str(cause)
        print "effect" + str(effect)
        for ii in range(0, len(cause)):
            write_str = write_str + " " + str(cause[ii])
        for jj in range(0, len(effect)):
            write_str = write_str + " " + str(effect[jj])
        print
        print "01 data，Granger causality test:"
        print "cause->effect"
        flag3 = False
        p_value_cause_to_effect3 = []
        ce3 = grangercausalitytests([[effect[i], cause[i]] for i in range(0, len(cause))], 5)
        for key in ce3:
            p_value_cause_to_effect3.append(ce3[key][0]["params_ftest"][1])
            if ce3[key][0]["params_ftest"][1] < 0.05:
                flag3 = True
        print "effect->cause"
        flag4 = False
        p_value_effect_to_cause4 = []
        ce4 = grangercausalitytests([[cause[i], effect[i]] for i in range(0, len(cause))], 5)
        for key in ce4:
            p_value_effect_to_cause4.append(ce4[key][0]["params_ftest"][1])
            if ce4[key][0]["params_ftest"][1] < 0.05:
                flag4 = True
        if flag3 and flag4:
            print "01 data，Granger two-way cause and effect"
            write_str = write_str + " " + "01 data，Granger two-way cause and effect"
            counter11_01 += 1
        elif flag3 and not flag4:
            print "01 data，Granger correct cause and effect"
            write_str = write_str + " " + "01 data，Granger correct cause and effect"
            counter10_01 += 1
        elif not flag3 and flag4:
            print "01 data，Granger wrong cause and effect"
            write_str = write_str + " " + "01 data，Granger wrong cause and effect"
            counter01_01 += 1
        elif not flag3 and not flag4:
            print "01 data，Granger no cause and effect"
            write_str = write_str + " " + "01 data，Granger no cause and effect"
            counter00_01 += 1
        write_str = write_str + " " + str(min(p_value_cause_to_effect3)) + " " + str(min(p_value_effect_to_cause4))
        print "01 data，CUTE causality test"
        delta_ce = bernoulli(effect) - cbernoulli(effect, cause)
        delta_ec = bernoulli(cause) - cbernoulli(cause, effect)
        if delta_ce > delta_ec and delta_ce - delta_ec >= -math.log(0.05, 2):
            print "01 data，CUTE correct cause and effect"
            write_str = write_str + " " + "01 data，CUTE correct cause and effect"
            counter_true += 1
        elif delta_ec > delta_ce and delta_ec - delta_ce >= -math.log(0.05, 2):
            print "01 data，CUTE wrong cause and effect"
            write_str = write_str + " " + "01 data，CUTE wrong cause and effect"
            counter_false += 1
        else:
            print "01 data，CUTE undecided"
            write_str = write_str + " " + "01 data，CUTE undecided"
            counter_undecided += 1
        write_str = write_str + " " + str(pow(2, -abs(delta_ce - delta_ec)))
        f.write(write_str)
        f.write("\n")
    print "-----------------"
    print "01 data，Granger causality test："
    print "two-way cause and effect:" + str(counter11_01)
    print "correct cause and effect:" + str(counter10_01)
    print "wrong cause and effect:" + str(counter01_01)
    print "no cause and effect:" + str(counter00_01)
    print "-----------------"
    print "01数据，CUTE causality test："
    print "correct cause and effect" + str(counter_true)
    print "wrong cause and effect" + str(counter_false)
    print "undecided:" + str(counter_undecided)


test_data()
test_random_zero_one_data()
add_noise_test()
