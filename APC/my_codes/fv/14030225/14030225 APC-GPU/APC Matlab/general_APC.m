function Recived_profile = general_APC(s, y_noisy, N, alpha, sigma)

M = length(alpha); % Number of Iteration
y_noisy = [zeros(M*(N-1),1);y_noisy;zeros(M*(N-1),1)];

%%%%%%%%%% Calculate Henkel Matrix 'S' %%%%%%%%%%
temp = s;
S = zeros(N,2*N-1);
for i = N:2*N-1
    S(:,i) = temp;
    temp = [0;temp(1:end-1)];
end
temp = s;
for i = N-1:-1:1
    temp = [temp(2:end);0];
    S(:,i) = temp;
end

%%%%%%%%%% Calculate Initial Stage %%%%%%%%%%
sum_s = zeros(N,N);
for i=1:2*N-1
    sum_s = sum_s + S(:,i)*S(:,i)';
end
W_int = sum_s^-1*s; % Initial W(l)

%%%%%%%%%% Calculate X1(l) %%%%%%%%%%
temp_X = zeros(1,length(y_noisy)-(N-1));
for i= 1:length(y_noisy)-(N-1)
    temp_X(i) = W_int'*y_noisy(i:i+N-1);
end

 X = temp_X; % Estimated X
%  X = conv(s,y_noisy); % Estimated X
%  X=X(1:285);
%%%%%%%%%% Reiterative Algorithm Part %%%%%%%%%%%
for j=1:M
    temp_X = zeros(1,length(X));
    rho = abs(X).^alpha(j); % Calculate |p(l)|^2
    for i = N:length(X)-N+1
        % Calculation C(l)
        temp_rho = rho(i-N+1:i+N-1);
        C = zeros(N);
        for k=1:2*N-1
            C = temp_rho(k)*S(:,k)*S(:,k)' + C;
        end
        R = sigma*eye(N);   % Noise Covariance Matrix
        W = (C+R)\s*rho(i);    % W(l) for each l %%finv = (C+R)^-1; f2 = s*rho(i); W2 = finv*f2; W2 = W;
        %%%%%%%%%% Calculate X(l) for Next Step %%%%%%%%%%
        if j == 1
            temp_X(i-N+1) = W'*y_noisy(i:i+N-1);
        else
            temp_X(i-N+1) = W'*y_noisy(i+(N-1)*(j-1):(j-1)*(N-1)+i+N-1);
        end
    end
    X = temp_X(1:length(X)-2*N+2);
%     hold on
%     grid on
end

Recived_profile = X;