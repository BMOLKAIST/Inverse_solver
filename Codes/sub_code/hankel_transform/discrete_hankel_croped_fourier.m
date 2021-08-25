classdef discrete_hankel_croped_fourier < handle
    properties (SetAccess = protected, Hidden = true)
        C;
        c;
        R;
        K;
        I;
        I2;
        
        k;
        r;
        n;
        
        use_GPU;
        
        pos;
    end
    methods
        function h=discrete_hankel_croped_fourier(max_r,samples,order,cropping,use_GPU)
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
            %find the cropping place
            [~,h.pos] = min(abs(h.k(:)-(max(h.k(:)).*cropping)));
            %compute other matrices
            %h.I=abs(besselj(1+h.n,h.c));
            h.I=abs(besselj(1+h.n,h.c));
            h.K=2*pi*h.R/h.C*h.I(:);
            h.R=h.I(:)/h.R;
            h.I=sqrt(2/h.C)./h.I;
            bess_coef=besselj(h.n,h.c(1:h.pos)'/h.C.*h.c);
            bess_coef2=besselj(h.n,h.c(:)/h.C.*h.c(1:h.pos));
            h.I2=h.I(:).*h.I(1:h.pos).*bess_coef2;
            h.I=h.I(1:h.pos)'.*h.I.*bess_coef;
            
            h.k=h.k(1:h.pos);
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
            vals_out=h.I*(vals./h.R).*h.K(1:h.pos);
        end
        function vals_out=backward(h,vals)
            if h.use_GPU
                vals=gpuArray(vals);
            end
            vals_out=h.I2*(vals./h.K(1:h.pos)).*h.R;
        end
    end
end