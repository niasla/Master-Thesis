clear all;
run('../HMMdata.m');

[p,path]=viterbi(T,B,P,{'b','a','c','c','d'},{'a','b','c','d'})