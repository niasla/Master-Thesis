clear all;
run('../HMMdata.m');

[p , alpha]=forward(T,B,P,{'b','a','c','c','d'},{'a','b','c','d'})