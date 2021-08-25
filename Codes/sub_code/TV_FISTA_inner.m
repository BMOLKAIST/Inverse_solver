function out_mat=TV_FISTA_inner(in_mat,lambda, non_neg,dirichlet_boundary,inner_itt,use_gpu)
% TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
%                     params.nmin, params.nmax, params.kappamax, h.parameters.use_non_negativity,dirichlet_boundary,inner_itt,use_gpu);
if use_gpu
    in_mat=single(gpuArray(in_mat));
else
    in_mat=single(in_mat);
end
    
    dim_num=length(size(in_mat));
    
    if dim_num>=3
        dividend=12;
    else
        dividend=8;
    end
    
    P_n=0;
    P_np{1}=0.*in_mat;
    P_np{2}=0.*in_mat;
    if dim_num==3
        P_np{3}=0.*in_mat;
    end
    R=P_np;
    
    t_n=1;
    t_np=1;
    
    %start the itterations
    for mm=1:inner_itt
        P_n=P_np;
        t_n=t_np;
        
        P_np=TV_L_trans(project_non_neg(in_mat-lambda*TV_L(R),non_neg,dirichlet_boundary));
        for kk=1:dim_num
            P_np{kk}=R{kk}+1/(dividend*lambda)*P_np{kk};
        end
        P_np=project_TV(P_np);
        
        t_np=(1+sqrt(1+4*t_n^2))/2;
        for kk=1:dim_num
            R{kk} = P_np{kk} + ((t_n-1)/t_np)*(P_np{kk}-P_n{kk});
        end
        
    end
    
    out_mat=project_non_neg(in_mat-lambda*TV_L(P_np),non_neg,dirichlet_boundary);
    
end

%% helping functions

function in_mat=project_non_neg(in_mat,non_neg,dirichlet_boundary)
if non_neg
    in_mat(real(in_mat)<0)=0;
end
in_mat(imag(in_mat)<0)=real(in_mat(imag(in_mat)<0));

if dirichlet_boundary
    in_mat(1,:,:)=0;
    in_mat(end,:,:)=0;
    in_mat(:,1,:)=0;
    in_mat(:,end,:)=0;
    in_mat(:,:,1)=0;
    in_mat(:,:,end)=0;
end
end

function P=project_TV(P)
if length(size(P{1}))==3
    A=sqrt(max(abs(P{1}).^2+abs(P{2}).^2+abs(P{3}).^2,1));
    P{1}=P{1}./A;
    P{2}=P{2}./A;
    P{3}=P{3}./A;
else
    A=sqrt(max(abs(P{1}).^2+abs(P{2}).^2,1));
    P{1}=P{1}./A;
    P{2}=P{2}./A;
end
end

function val=TV_val(in_mat)
P=TV_L_trans(in_mat);
if length(size(in_mat))==3
    val=sum(sqrt(abs(P{1}).^2+abs(P{2}).^2+abs(P{3}).^2),'all');%iso tv
else
    val=sum(sqrt(abs(P{1}).^2+abs(P{2}).^2),'all');%iso tv
end
end

function out_mat=TV_L(P)
out_mat=P{1}+P{2};
if length(size(out_mat))==3
    out_mat=out_mat+P{3};
end

if length(size(out_mat))==3
    out_mat=out_mat-circshift(P{1},[1 0 0]);
    out_mat=out_mat-circshift(P{2},[0 1 0]);
    out_mat=out_mat-circshift(P{3},[0 0 1]);
else
    out_mat=out_mat-circshift(P{1},[1 0]);
    out_mat=out_mat-circshift(P{2},[0 1]);
end
end
function out_mat=TV_L_trans(in_mat)
if length(size(in_mat))==3
    out_mat{1}=in_mat-circshift(in_mat,[-1 0 0]);
    out_mat{2}=in_mat-circshift(in_mat,[0 -1 0]);
    out_mat{3}=in_mat-circshift(in_mat,[0 0 -1]);
else
    out_mat{1}=in_mat-circshift(in_mat,[-1 0]);
    out_mat{2}=in_mat-circshift(in_mat,[0 -1]);
end
end

