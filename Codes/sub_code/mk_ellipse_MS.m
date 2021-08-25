
function H = mk_ellipse_MS(varargin)
if length(varargin)==4
    XR=varargin{1};YR=varargin{2};
    X=varargin{3};Y=varargin{4};
    [XX, YY]=meshgrid(1:X,1:Y);
    H=((XX-floor(X/2)-1)./XR).^2+((YY-floor(Y/2)-1)./YR).^2>1.0;
elseif length(varargin)==6
    XR=varargin{1};YR=varargin{2};ZR=varargin{3};
    X=varargin{4};Y=varargin{5};Z=varargin{6};
    [XX, YY,ZZ]=meshgrid(1:X,1:Y,1:Z);
    H=((XX-floor(X/2)-1)./XR).^2+((YY-floor(Y/2)-1)./YR).^2+((ZZ-floor(Z/2)-1)./ZR).^2>1.0;
else
    error('즐')
end
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end