function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

visible_data = sample_bernoulli(visible_data);

% 1 obtain hidden state
hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_data);
%rbm_w = rbm_w + d_G_by_rbm_w;
% 2 sample from hidden probability
hidden_state = sample_bernoulli(hidden_probability);  %convert probability to binary state(>0.5)
d_G_by_rbm_w0 = configuration_goodness_gradient(visible_data, hidden_state);

% 3 reconstruct visible from hidden
visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
visible_state = sample_bernoulli(visible_probability);  %convert probability to binary state(>0.5)
hidden_probability2 = visible_state_to_hidden_probabilities(rbm_w, visible_state);
%hidden_state2 = sample_bernoulli(hidden_probability2);  %convert probability to binary state(>0.5)

% 4 obtain the gradient .
%  d_G_by_rbm_w1 = configuration_goodness_gradient(visible_state,
%  hidden_state2);  % Q7

d_G_by_rbm_w1 = configuration_goodness_gradient(visible_state, hidden_probability2);
ret = d_G_by_rbm_w0 -d_G_by_rbm_w1;
%error('not yet implemented');
end
