%% A framework of investigating sparse coding, predictive coding in temporal domain
% After learning, basis vectors should resembel Gabor like features.

clc
clear;
close all;

%% SP parameters
sp.dt = 3; % unit: ms; time step of the simulation
sp.delta_u = 0; % unit: ms; neural delay in feedforward pathway
sp.delta_d = 0; % unit: ms; neural delay in feedback pathway

sp.n_t_infer = 100; % number of iterations of inference
sp.n_epoch = 3e4;
sp.n_t = sp.n_epoch * sp.n_t_infer; % number of time steps of one epoch
sp.n_t_input = sp.n_t_infer; % duration of presentation of one input
sp.n_t_display = 10 * sp.n_t_infer; % frequency of display infer

sp.n_delta_u = sp.delta_u / sp.dt;
sp.n_delta_d = sp.delta_d / sp.dt;

sp.eta_A = 1e0; % learning rate of A
sp.eta_S = 3e-2; % update rate of response
sp.beta = 0e-2; % constant for weight penalty

sp.sz = 16; % Size of the image patch
sp.L = sp.sz^2; % Length of the input vector
sp.M = 2*sp.L; % Number of neurons in the network
sp.batch_size = 100; % batch training

sp.s_max = 30; % upper bound of response
sp.s_min = 0;

sp.A = normalize_matrix(randn(sp.L, sp.M)); % Connections between input and output layer

%% Homeostasis function parameters
h.lambda = 0.5;
h.n = 4;
h.x0 = 0.5;
h.sigma = h.x0/1;

S = 0 : 0.001 : 1;

H_s = h.lambda * ( double(S<h.x0).*(2*(S/h.sigma)./(1+(S/h.sigma).^2)) + ...
    double(S>h.x0).*(exp(((S-h.x0)/h.sigma).^h.n)) + ...
    double(S==h.x0) );

figure(1);
plot(S, H_s, 'linewidth', 2);
xlabel('s');
ylabel('H(s)')
grid on
set(gca,'fontsize',16)

%% Definitions of symbols
I = zeros(sp.L, sp.batch_size); % Input matrix: each column is an image patch in the vector form (sz*sz, 1)
S = zeros(sp.M, sp.batch_size, 1+sp.n_delta_d); % Firing rates (Response) of M neurons for 1 input
E = zeros(sp.L, sp.batch_size, 1+sp.n_delta_u); % initialisation [t, t-delta_u, ..., t-2, t-1]
S_history = zeros(sp.n_t_display,sp.M); % History of S

%% Load image
sp.training_dataset = 'IMAGES_SparseCoding.mat';
load(sp.training_dataset) % The dataset with 10 pre-whitened natural images

num_images = size(IMAGES_WHITENED,3); % Number of images in the data set
image_size = size(IMAGES_WHITENED,1); % Size of the images
IMAGES = IMAGES_WHITENED; clear IMAGES_WHITENED
BUFF = 4; % Buffer size that exclude image patches close to boundaries

%% Display A and S
fig = figure(1);fig.Units = 'normalized'; fig.Position = [0 0 0.4 0.8];

figure(1);
subplot(2,2,1);display_matrix(sp.A,3); title('A'); colormap(gray);
subplot(4,2,5);stem(S(:,1,1));title(['S: firing rates']); xlabel('cell ID');
subplot(4,2,6);plot(S_history);
xlabel('i_t');title('S: response trajectories')

%% main loop
for i_t = 1 : sp.n_t
    %% prensent input I
    if mod(i_t, sp.n_t_input) == 1
        % Randomly pick an image
        i = ceil(num_images*rand);
        this_image = IMAGES(:,:,i);

        % Randomly extract image patchesfrom this image to make data vector I
        for i = 1 : sp.batch_size
            % Choose the left-up point of the image
            r = BUFF + ceil((image_size-sp.sz-2*BUFF)*rand);
            c = BUFF + ceil((image_size-sp.sz-2*BUFF)*rand);

            % Shape the image patch into vector form (sz*sz, 1) where L = sz * sz
            X_data(:,i) = reshape(this_image(r:r+sp.sz-1,c:c+sp.sz-1), sp.L, 1);
        end
        I = X_data;
    end

    %% Inference stage
    % compute index for t and t-delay_u of E
    i_t_E = mod(i_t-1, 1+sp.n_delta_u)+1;
    i_t_E_delay_u = mod(i_t, 1+sp.n_delta_u)+1;

    % compute index for t and t-delay_d of S
    i_t_S = mod(i_t-1, 1+sp.n_delta_d)+1;
    i_t_S_delay_d = mod(i_t-1+1, 1+sp.n_delta_d)+1;

    % E(t)=I(t)-A*S(t-delta_d)
    E(:,:,i_t_E) = I - sp.A * S(:,:,i_t_S_delay_d);

    % compute Q'(s)
    H_S = h.lambda * ( double(S(:,:,i_t_S)<h.x0).*(2*(S(:,:,i_t_S)/h.sigma)./(1+(S(:,:,i_t_S)/h.sigma).^2)) + ...
        double(S(:,:,i_t_S)>h.x0).*(exp(((S(:,:,i_t_S)-h.x0)/h.sigma).^h.n)) + ...
        double(S(:,:,i_t_S)==h.x0) );


    % S(t+1) = S(t) + dt*eta*[ A'*E(t-delta_u) - lambda*Q'(S(t)) ]
    S(:,:,i_t_S_delay_d) = S(:,:,i_t_S) + sp.dt * sp.eta_S * ( sp.A'*E(:,:,i_t_E_delay_u) - H_S );
    S(:,:,i_t_S_delay_d) = min(max(S(:,:,i_t_S_delay_d), sp.s_min), sp.s_max);

    S_history(mod(i_t-1,sp.n_t_display)+1,:) = S(:,1,i_t_S_delay_d); % Record the response history of S

    %% Learning stage: after n_t_infer cycle of inference
    if mod(i_t, sp.n_t_infer) == 0
        if max(S(:,i_t_S_delay_d)) == 0
        else
            A_past = sp.A;
            dA = E(:,:,i_t_E) * S(:,:,i_t_S)' / sp.batch_size - sp.beta * sp.A;
            sp.A = sp.A + sp.eta_A * dA;
            sp.A = normalize_matrix(sp.A, 'L2 norm', 1); % Normalize each column of the connection matrix
        end
    end

    %% Display plots
    if (mod(i_t,sp.n_t_display) == 0)
        figure(1);
        subplot(2,2,1);display_matrix(sp.A,3); title('A'); colormap(gray);
        subplot(2,2,2);display_matrix(I,3); axis square; title('Input image patches'); colormap(gray);
        subplot(4,2,5);stem(S(:,1, 1));title('S: firing rates for the first input'); xlabel('cell ID');
        subplot(4,2,6);plot(S_history);
        xlabel('i_t');title('S: response trajectories')
    end

end

%% save results
save('results.mat', 'sp', 'h');



































