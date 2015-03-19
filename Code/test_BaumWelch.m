clear all;
run('../HMMdata.m');

[At, Bt, pt] = HMMtrain(T, B, P, {'b','a','c','c','d'}, {'a','b','c','d'} ,1)