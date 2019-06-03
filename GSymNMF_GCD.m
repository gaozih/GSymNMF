function [H, H1, objGCD, timeGCD] = GSymNMF_GCD(V, lamda, miu, k, maxiter, Hinit, tra)
% Graph Symmetric Nonnegative Matrix Factorization (NMF) via Greedy Coordinate Descent
%
% Usage: [W H objGCD timeGCD] = GSymNMF_GCD(V, k, maxiter, Winit, Hinit, trace)
%
% Author: Ziheng Gao
%
% input: 
%		V: the input m by m dense matrix
%               lambda, miu: parameter in the objective equation.
%		k: the specified rank
%		maxiter: maximum number of iterations
%		Winit: initial of W (m by k dense matrix)
%		Hinit: initial of H (k by m dense matrix)
%		trace: 1: compute objective value per iteration.
%			   0: do not compute objective value per iteration. (default)
%
% output: 
%		NMF_GCD will output nonnegative matrices W, H, such that WH is an approximation of V
%		W: m by k dense matrix
%		H: k by m dense matrix
%		objGCD: objective values. 
%		timeGCD: time taken by GCD. 
%

n = size(V,1);
%W = Winit;
S = V;
D = diag(sum(S,2));
L = D - S;
H = Hinit;
H1 = H;
%% Stopping tolerance for subproblems
tol = 0.01;
LAMDA = zeros(size(H));
total = 0;
obj = zeros([1,5]);
fprintf('Start running NMF_GCD with trace=%g\n', tra);
obj(1) =  0.5*norm(V-H*H1','fro')^2 + 0.5*lamda*trace(H'*L*H1) + trace(LAMDA'*(H-H1)) + 0.5*miu*norm(H-H1,'fro')^2;
obj(2) = 0.5*norm(V-H*H1','fro')^2;
obj(3) = 0.5*lamda*trace(H'*L*H1);
obj(4) = trace(LAMDA'*(H-H1));
obj(5) = 0.5*miu*norm(H-H1,'fro')^2;
fprintf('Iteration %g, objective value', 0);
fprintf(' %g,',obj);
fprintf('\n');
for iter = 1:maxiter
	begin = cputime;
    

	% update variables of H
	VH1 = V*H1;
	H1H1 = H1'*H1;
    LH1 = L * H1;
	GH = -VH1 + H*H1H1+ lamda*LH1+LAMDA+miu*(H-H1);
	
	Hnew = ggc_h(GH', H1H1,H',miu,tol, k^2); % Coordinate descent updates for H
	H = Hnew';

	% update variables of W
	VH = V*H;
	HH = H'*H; % Hessian of each row of W
	LH= L * H;
	%GW = -(VH - W*HH); % gradient of W
    GH1 = -VH + H1*HH+ lamda*LH-LAMDA+miu*(H1-H);
	H1new = ggc_h(GH1', HH,H1',miu,tol, k^2); % Coordinate descent updates for H
	H1 = H1new';
    
    LAMDA = LAMDA + (H - H1);

	total = total + cputime - begin;
	if tra==1
		obj(1) =  0.5*norm(V-H*H1','fro')^2 + 0.5*lamda*trace(H'*L*H1) + trace(LAMDA'*(H-H1)) + 0.5*miu*norm(H-H1,'fro')^2;
        obj(2) = 0.5*norm(V-H*H1','fro')^2;
        obj(3) = 0.5*lamda*trace(H'*L*H1);
        obj(4) = trace(LAMDA'*(H-H1));
        obj(5) = 0.5*miu*norm(H-H1,'fro')^2;
        
		objGCD(iter,:) = obj;
		timeGCD(iter) = total;
		if mod(iter,20)==0
			fprintf('Iteration %g, objective value', iter);
            fprintf(' %g,',obj);
            fprintf('\n');
		end
	end
end
if tra==0
	obj =  0.5*norm(V-H*H1','fro')^2 + 0.5*lamda*trace(H'*L*H1) + trace(LAMDA'*(H-H1)) + 0.5*miu*norm(H-H1,'fro')^2;
	objGCD = obj;
	timeGCD = total;
end
fprintf('Finished NMF_GCD with trace=%g, final objective value %g\n', tra,obj(1));
