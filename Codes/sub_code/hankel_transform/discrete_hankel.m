classdef discrete_hankel < handle
    properties (SetAccess = protected, Hidden = true)
        C;
        c;
        R;
        K;
        I;
        k;
        r;
        n;
        
        use_GPU;
    end
    methods
        function h=discrete_hankel(max_r,samples,order,use_GPU)
            h.use_GPU=use_GPU;
            h.n=single(order);
            h.R=single(max_r);
            roots=single(JnRoots(h.n,samples+1));
            if h.use_GPU
                h.n=gpuArray(h.n);
                h.R=gpuArray(h.R);
                roots=gpuArray(roots);
            end
            h.C=roots(end);
            h.c=roots(1:end-1);
            h.r=h.R/h.C*h.c(:);
            h.k=h.c(:)/h.R;
            h.I=abs(besselj(1+h.n,h.c));
            h.K=2*pi*h.R/h.C*h.I(:);
            h.R=h.I(:)/h.R;
            h.I=sqrt(2/h.C)./h.I;
            h.I=h.I(:)*h.I.*besselj(h.n,h.c(:)/h.C*h.c);
        end
        function r=get_r(h)
            r=h.r;
        end
        function k=get_k(h)
            k=h.k;
        end
        function vals_out=forward(h,vals)
            if h.use_GPU
                vals=gpuArray(vals);
            end
            vals_out=h.I*(vals./h.R).*h.K;
        end
        function vals_out=backward(h,vals)
            if h.use_GPU
                vals=gpuArray(vals);
            end
            vals_out=h.I*(vals./h.K).*h.R;
        end
    end
end