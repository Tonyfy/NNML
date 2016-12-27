function visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.

%according rbm weight and hidden unit state,calculate visible unit state.
visible_probability = 1./(1+exp(-1*rbm_w'*hidden_state));  %rbm_w --- L2xL1   hidden--- L2xC  visi --- L1xC
%error('not yet implemented');
end
