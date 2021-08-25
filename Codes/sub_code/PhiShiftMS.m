function [goodp2,mask, mdx, mdy,height]=PhiShiftMS(varargin)
    % PhiShiftMS
        p2=varargin{1};
        if length(varargin)==1
            maskstyle=1;
        elseif length(varargin)==2
            maskstyle=varargin{2};
            n=1;
        elseif length(varargin)==3
            maskstyle=varargin{2};
            n=varargin{3};
        else
            maskstyle=varargin{2};
            n=varargin{3};
            input_mask=varargin{4};
        end
        
        [imY, imX]=size(p2);
%         Uimg=unwrap2(Uimg);clc;
        bsize=8;
              
            
        switch maskstyle
            case 0 %Conserve image   
                goodp2=p2;mdx=0;mdy=0;mask=0;
            case 1 %No BGmask
                mask=ones(imY,imX);
                mask(bsize:imY-bsize+1,bsize:imX-bsize+1)=0;
                switch length(varargin)
                    case 2
                        [goodp2,coefficients]=D2_LSAms(p2,1,mask);
                         mdx=coefficients(1);mdy=coefficients(2);height=coefficients(3);
                    case 3
                        [goodp2,coefficients]=D2_LSAms(p2,n,mask);
                        if n~=0
                        mdx=coefficients(n);mdy=coefficients(2*n);height=coefficients(2*n+1);
                        else
                           mdx=0;mdy=0;height=coefficients; 
                        end
                    case 4
                        mask=input_mask;
                        [goodp2,coefficients]=D2_LSAms(p2,n,mask);
                        if n~=0
                        mdx=coefficients(n);mdy=coefficients(2*n);height=coefficients(2*n+1);
                        else
                           mdx=0;mdy=0;height=coefficients; 
                        end
                end

            case 2 %selected BGmask until modified image is satisfactory
                while true
                    figure;imagesc(p2);colorbar;
                    vertices=ginput();
                    close;
                    px_s=vertices(:,1)'; py_s=vertices(:,2)';
                    mask=poly2mask(px_s,py_s,imY,imX);
                    switch nargin
                        case 3
                            [goodp2,coefficients]=D2_LSAms(p2,1,mask);
                            mdx=coefficients(1);mdy=coefficients(2);height=coefficients(3);
                        case 4
                            [goodp2,coefficients]=D2_LSAms(p2,n,mask);
                            mdx=coefficients(n);mdy=coefficients(2*n);height=coefficients(2*n+1);

                    end

                      figure;imagesc(goodp2);colorbar;
                      answer=input('satisfied?', 's');
                  
                    if strcmp(answer,'y')
                        close;
                        break;
                    end
                    close;
                end
            case 3 %selected BGmask2 until modified image is satisfactory
                while true
                    figure;imagesc(p2);colorbar;
                    vertices=ginput();
                    close;
                    px_s=vertices(:,1)'; py_s=vertices(:,2)';
                    mask=~poly2mask(px_s,py_s,imY,imX);
                    switch nargin
                        case 3
                            [goodp2,coefficients]=D2_LSAms(p2,1,mask);
                            mdx=coefficients(1);mdy=coefficients(2);height=coefficients(3);
                        case 4
                            [goodp2,coefficients]=D2_LSAms(p2,n,mask);
                            mdx=coefficients(n);mdy=coefficients(2*n);height=coefficients(2*n+1);

                    end

                      figure;imagesc(goodp2);colorbar;
                      answer=input('satisfied?', 's');
                  
                    if strcmp(answer,'y')
                        close;
                        break;
                    end
                    close;
                end
            case -1
                goodp2=p2.*(-1);mdx=0;mdy=0;mask=0;
        end
end           
function [goodp2,coefficients]=D2_LSAms(varargin)
    % Nth order ramp elimination based upon least square approximation
    % coefficient: [xn, xn-1,...x1,yn,yn-1,...y1,a0]
    % Except 0-padded map
        p2=varargin{1};
        if length(varargin)==1
            n=1;
        end
        
        switch length(varargin)
            case 2        
                n=varargin{2};
                if n~=0
                    X=zeros(imY*imX,n);Y=X;
                    for ii=1:n
                        XXX=XX.^ii;X(1:imY*imX,ii)=XXX(:);
                        YYY=YY.^ii;Y(1:imY*imX,ii)=YYY(:);
                    end
                    E=ones(imY*imX,1);AA=[X,Y,E];
                    coefficients=(AA'*AA)\(AA'*p2(:));
                else
                    coefficients=mean2(p2);
                end

            case 3
                n=varargin{2};
                mask=varargin{3};
                
                [imY, imX]=size(p2);
                [XX, YY]=meshgrid(1:imX,1:imY);

                p2mask=p2.*mask;
                p2mask=omit_outliers(p2mask);
                p2mask=p2mask(:);     
                if n~=0
                X=zeros(sum(p2mask~=0),n);Y=X;
                    for ii=1:n
                    XXX=XX.^ii;XXX=XXX(:);XXX(p2mask==0)=[];X(:,ii)=XXX;
                    YYY=YY.^ii;YYY=YYY(:);YYY(p2mask==0)=[];Y(:,ii)=YYY;
                    end
                    p2mask(p2mask==0)=[];
                    E=ones(sum(p2mask~=0),1);AA=[X,Y,E];
                    coefficients=(AA'*AA)\(AA'*p2mask);
                else
                   coefficients=sum(sum(p2mask))./sum(sum(mask)); 
                end
        end
        goodp2=p2-coefficients(end).*ones(imY,imX);
        for ii=1:n
           goodp2=goodp2-coefficients(ii).*XX.^ii;
           goodp2=goodp2-coefficients(n+ii).*YY.^ii;
        end
        
        
end
function p2mask=omit_outliers(p2mask)
    list=p2mask(:);list(list==0)=[];
    p25=prctile(double(list),25);p75=prctile(double(list),75);
    cmin=p25-1.5*(p75-p25);cmax=p75+1.5*(p75-p25);
    p2mask(p2mask<cmin)=0;
    p2mask(p2mask>cmax)=0;
end