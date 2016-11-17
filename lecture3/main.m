% data1 = load('Datasets/dataset1.mat');
% neg_example = data1.neg_examples_nobias;
% pos_eaxmple = data1.pos_examples_nobias;
% w_init = data1.w_init;
% w_gen_feas = data1.w_gen_feas;
% 
% W = learn_perceptron(neg_example,pos_eaxmple,w_init,w_gen_feas);
% 
% [mistakes0, mistakes1] =  eval_perceptron(neg_example, pos_example, w);
% 
% plot_perceptron(neg_example,pos_eaxmple,mistakes0,mistakes1,W);

load('Datasets/dataset1.mat')
% load('Datasets/dataset2.mat')
% load('Datasets/dataset3.mat')
% load('Datasets/dataset4.mat')
w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas);