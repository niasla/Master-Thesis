clear all;
run('../HMMdata.m');

[p,beta]=backward(T,B,P,{'b','a','c','c','d'},{'a','b','c','d'})