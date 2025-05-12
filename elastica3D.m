function [x3,y3,z3,F,P,slip,varargout] = elastica3D(x,y,z,C,varargin)
%% function [x3,y3,z3,F,P] = elastica3D(x,y,z,C,varargin)
% -------------------------------------------------------------------------
% Three-dimensional elastic beam model
% Lucie Huet: 2011-2013 
% Initially based on elastica2D by Joe Solomon: 2005-2006
% Email for correspondance: Mitra Hartmann- hartmann@northwestern.edu
%
% OVERVIEW:
% The beam model has three modes of operation:
%   (1) 'force' - a force (f) is applied at an arc-length distance (s)
%           along the beam length and at an angle zeta (z)
%   (2) 'point' {default} - the model varies f,s and z until the beam is
%           deflected to pass through a contact point at [a,b,c]
%   (3) 'peg' - the model varies f,s, and z until the beam is deflected to
%           pass through a frictionless peg defined by {[peg_origin]
%           [peg_direction]}
% -------------------------------------------------------------------------
% INPUTS:
%   (x,y,z) - undeflected beam shape [m]
%   C - beam bending constraint depending on the mode:
%       'force': [f,s,z] Force (f)[N] applied at arc length (s)[m] and at
%           angle zeta (z)[radians]
%       'point' {default}: [a,b,c] point location in 3D space [m]
%       'peg': {[a,b,c] [i,j,k]}  [a,b,c] defines the point location of
%           origin in 3D space for the peg [m], and [i,j,k] define the unit
%           vector describing the pegs orientation. 
%
% EXTRA PARAMETER INPUTS:
%   + entered under varargin, structed as ...,'param_name',param,..
%   // MODEL
%       'mode'  - switch between modes
%           + 'force': apply [f,s,z]
%           + 'point' {default}: find [f,s,z] to deflect to point [a,b,c]
%           + 'peg': find [f,s,z] to deflect to pass through frictionless
%             peg defined by {[peg_origin] [peg_direction]}
%   // PLOTTING
%       'plot'  - plot final result (true / false {default})
%   // BEAM PARAMETERS
%       'E'     - Elastic/Young's modulus [Pa]
%               + {default}: Constant of 3.3 GPa for each node
%               + # : constant value for each node
%               + [1xN]: specify modulus at each node
%       'I'     - Area moment of inertia [m^4]
%               + {default}: Tapered bean with taper ratio of 15 and base
%                 diameter of 100 um
%               + # : constant value for each node
%               + [1xN]: specify area moment of inertia at each node
%       'G'     - Shear modulus [Pa]
%               + {default}: Based off of Young's modulus using Poisson's
%                 ratio of v = 0.38 [A]
%               + # : constant value for each node
%               + [1xN]: specify shear modulus at each node
%       'J'     - Torsion Constant [m^4]
%               + {default}: Based off Area mmoment of inertia, J = 2*I
%               + # : constant value for each node
%               + [1xN]: specify torsion constant at each node
%       'BC'    - Change the boundary conditions at the first node
%               + 'r': rigid boundary condition (E(1) and G(1) set to Inf)
%               + 'e' {default}: elastic boundary condition
%               + [# #]: elastic boundary condition with input as stiffness
%                 [EI GJ] at first node
%   // OPTIMIZATION SETTINGS
%       'opt-params' - optimization parameters for fminsearch, input as
%                      cell array
%       'fsz_guess'  - provide an initial guess for [f,s,z]
% OUTPUTS:
%   (x3,y3,z3) - deflected beam shape (3D!) [m]
%   F - struct of force information
%       .fsz - [f s z] of force
%       .f - [3x1] force vector [N]
%       .m - [3x1] moment vector around whisker base [N m]
%       .pos - (x,y,z) location where force is applied on deflected beam
%              [m]
%   P - struct of whisker parameters
%       .dth - [1x(N-1)] theta angle change between whisker nodes [radians]
%       .dphi - [1x(N-1)] phi angle change between whisker nodes [radians]
%       .ds - [1x(N-1)] length of each link [m]
%       .s - [1xN] arc length at each node, starting at 0 [m]
%       .E - [1xN] Young's modulus at each node [Pa]
%       .I - [1xN] Area moment of inertia at each node [m^4]
%       .G - [1xN] Shear modulus at each node [Pa]
%       .J - [1xN] Torsion constant at each node [m^4]
% -------------------------------------------------------------------------
% MODEL ASSUMPTIONS
% + The simulation runs on the following assumptions:
%   {1} The whisker base is centered at (0,0)
%   {2} The first whisker segment is co-linear with the x-axis and pointing
%       in the positive x direction
%   {3} The whisker has a circular cross-section
%   {4} There is no friction, so the force is applied normal to the whisker
%       surface
% + The resulting forces/moments at the base should be thought of as "what
%   the whisker applies to the follicle"
% -------------------------------------------------------------------------
% EXAMPLES:
%   [x3,y3,z3] = elastica3D(x,y,z,[f,s,z],'mode','force');
%   [x3,y3,z3] = elastica3D(x,y,z,[a,b,c],'E',E,'I',I);
%   [x3,y3,z3] = elastica3D(x,y,z,[a,b,c],'opt_params',{'TolX',1e-12})
% -------------------------------------------------------------------------
% Citations: 
% [A] Etnier, S. A. (2003). Twisting and bending of biological beams: 
%       Distribution of biological beams in a stiffness mechanospace. The 
%       Biological Bulletin 205, 36-46.
% [B] Huet, L. A. and Hartmann, M. J. Z. (2016). Simulations of a vibrissa 
%       slipping along a straight edge and an analysis of frictional 
%       effects during whisking. IEEE Transactions on Haptics 9, 158-169.
% [C] Huet, L. A., Rudnicki, J. W. and Hartmann, M. J. Z. (2017). Tactile 
%       sensing with whiskers of various shapes: Determining the three-
%       dimensional location of object contact based on mechanical signals 
%       at the whisker base. Soft Robotics 4, 88-102.
% [D] Huet, L. A., Schroeder, C. L. and Hartmann, M. J. Z. (2015). Tactile 
%       signals transmitted by the vibrissa during active whisking 
%       behavior. Journal of Neurophysiology 113, 3511-3518.
% [E] Yang, A. E. T. and Hartmann, M. J. Z. (2016). Whisking kinematics 
%       enables object localization in head-centered coordinates based on 
%       tactile information from a single vibrissa. Frontiers in Behavioral 
%       Neuroscience 10, 145.
% [F] Quist, B. W., Faruqi, R. A. and Hartmann, M. J. Z. (2011). Variation 
%       in Young's modulus along the length of a rat vibrissa. Journal of 
%       Biomechanics 44, 2775-2781.
% [G] Hires, S. A., Pammer, L., Svoboda, K. and Golomb, D. (2013). Tapered 
%       whiskers are required for active tactile sensation. eLife 2, 
%       e01350.
% [H] Williams, C. M. and Kramer, E. M. (2010). The advantages of a 
%       tapered whisker. PLoS One 5, e8806.
% [I] Belli, H. M., Bresee, C. S., Graff, M. M. and Hartmann, M. J. Z. 
%       (2018). Quantifying the three-dimensional facial morphology of the 
%       laboratory rat with a focus on the vibrissae. PLoS One 13, 
%       e0194981.
%% INITIALIZE
global LOCAL

if nargin<1
    npts = 100;
    [x,y,z] = Get_RatMap('LC4','Npts',npts);
    x = x./1000;
    y = y./1000;
    z = z./1000;
    [x,y,z] = equidist3D(x,y,z);
    
    r_base = 0.0001/2; %100um base diameter
    taper = 15;
    r = linspace(r_base,r_base/taper,npts);
    I_in = (pi/4)*r.^4;
    
    varargin = {'mode','point','plot',1,'I',I_in,'plot_steps',0,'BC','r'}; 

end

%% PARAMETER DEFAULTS

% Plotting:
TGL_plotfinal = 0; % Plot final deflected shape
TGL_plotsteps = 0; % Plot each iteration of the fitting algorithm

% Mode of operation:
TGL_mode = 'point'; % 'force' or 'point' or 'peg'

% Optimization parameters:
opt_params = {'TolX',1e-17,'MaxFunEvals',10000}; %5000};
fsz_guess = []; % [f s z] ... units of:[N m rad]
% ds_thresh = 1e-10; % [m]

% Whisker parameters:
TGL_BC = 'e'; % Toggle boundary condition: 'r','e',or [EI GJ]
E = NaN; % [Pa]
I = NaN; % [m^4]
G = NaN; % [Pa]
J = NaN; % [m^4]
P = struct; % initialize empty whisker parameter struct

%% USER INPUT PARAMETERS
slip=0;
for ii = 1:2:length(varargin)
    switch varargin{ii}
        case 'plot',        TGL_plotfinal = varargin{ii+1};
        case 'plot_steps',  TGL_plotsteps = varargin{ii+1};
        case 'mode',        TGL_mode = varargin{ii+1};
        case 'BC',          TGL_BC = varargin{ii+1};
        case 'E',           E = varargin{ii+1};
        case 'I',           I = varargin{ii+1};
        case 'G',           G = varargin{ii+1};
        case 'J',           J = varargin{ii+1};
        case 'opt_params',  opt_params = [opt_params varargin{ii+1}]; %#ok<AGROW>
        case 'fsz_guess',   fsz_guess = varargin{ii+1};
        otherwise
            warning('elastica3D:BadInput',[varargin{ii},' not supported - Ignoring']);
    end
end

%% SETUP WHISKER GEOMETRY

% Make (x,y,z) rows instead of columns
if size(x,1) ~= 1, x = x'; end
if size(y,1) ~= 1, y = y'; end
if size(z,1) ~= 1, z = z'; end
x1 = x; y1 = y; z1 = z;

% put into one matrix W
W = [x1;y1;z1];
% initiate matrix sizes
len = length(x1)-1;
dth = zeros(1,len);
dphi = dth;

for ii = 1:len
    % bring next node to origin
    W(1,:) = W(1,:) - W(1,ii);
    W(2,:) = W(2,:) - W(2,ii);
    W(3,:) = W(3,:) - W(3,ii);
    
    dth(ii) = atan2(W(2,ii+1),W(1,ii+1)); % z-axis rotation in xy plane
    W = getRZ(-dth(ii))*W; % rotate whisker
    
    dphi(ii) = atan2(W(3,ii+1),W(1,ii+1)); % y-axis rotation away from z axis
    W = getRY(dphi(ii))*W; % rotate whisker, yes negative
end

% get all real arc lengths
ds = sqrt(diff(x1).^2+diff(y1).^2+diff(z1).^2);
s = [0 cumsum(ds)];

% put results into whisker parameter struct
P.dth = dth;
P.dphi = dphi;
P.ds = ds;
P.s = s;

clear W len dth dphi ds s

%% SETUP WHISKER PARAMETERS

% Setup: E
% E value selected from: Quist et al. (2012) "Variation in Young's modulus
%   along the length of a rat vibrissa." J. Biomechanics. 44:2775-2781
if isnan(E(1)),
    P.E = (3.3e9).*ones(1,length(x1));
elseif length(E) == 1,
    P.E = E.*ones(1,length(x1));
elseif length(E) == length(x1),
    P.E = E;
else
    error('length of E is not length(1) or length(x)');
end
if size(P.E,2) == 1, P.E = P.E'; end

% Setup: I
if isnan(I(1)), % No I provided.
    % Assume: linear taper with a taper ratio of 15.
    % Assume: base radius of 100 um (typical for a large caudal vibrissa)
    % taper ratio = base diameter / tip diameter
    taper = 1/15;
    r = (0.00255/2).*linspace(1,taper,length(x1));
    P.I = (0.25*pi).*(r.^4);
    clear taper r
elseif length(I) == 1, % Assume cylinder
    P.I = I.*ones(1,length(x1));
elseif length(I) == length(x1),
    P.I = I;
else
    error('length of I is not 1 or length(x)');
end
if size(P.I,2) == 1, P.I = P.I'; end

% Setup: G (Shear Modulus)
% Base off E value and Poisson's ratio selected from:
% Hu et al. (2010) "Measurement of Young's modulus and Poisson's ratio of
% Human Hiar using Optical techniques."  SPIE
% Etnier (2003) "Twisting and Bending of Biological Beams:: Distribution of
% Biological Beams in a Stiffness Mechanospace." Marine Biological
% Laboratory
% Zhao et al. (2009) "Experimental study on the mechanical properties of
% the horn sheaths from cattle." J. of Experimental Biology
if isnan(G(1))
    v = 0.38;
    P.G = P.E./(2*(1+v));
    clear v
elseif length(G) == 1
    P.G = G.*ones(1,length(x1));
elseif length(G) == length(x1)
    P.G = G;
else
    error('length of G is not 1 or length(x)');
end
if size(P.G,2) == 1, P.G = P.G'; end;

% Setup: J
if isnan(J(1))
    % Assuming circular cross section, J = 2*I
    P.J = P.I.*2;
elseif length(J) == 1
    P.J = J.*ones(1,length(x1));
elseif length(J) == length(x1)
    P.J = J;
else
    error('length of J is not 1 or length(x)');
end
if size(P.J,2) == 1, P.J = P.J'; end;

% Setup: BC effect
% create rigid matrix (for avoiding if statement in loop later)
switch TGL_BC
    case 'e', % Do nothing
    case 'r'
        % make E and I at base infinity to prevent bending
        P.E(1) = Inf;
        P.G(1) = Inf;
    otherwise
        % EI and/or GJ specified to adjust BC
        if ~isnan(TGL_BC(1))
            % EI defined, change E
            P.E(1) = TGL_BC(1)/P.I(1);
        end
        if ~isnan(TGL_BC(2))
            % GJ defined, change G
            P.G(1) = TGL_BC(1)/P.J(1);
        end
end

%% MODE SPECIFIC PROCESSES AND SHAPE CALCULATION

switch TGL_mode
    case 'force'
        
        f_mag = C{1}(1);
        s_force = C{1}(2);
        zeta = C{1}(3);
        
        LOCAL_ComputeShape_3D([1,1,1],[f_mag s_force zeta],...
            x1,y1,z1,P,TGL_mode,C,TGL_plotsteps);
        % LOCAL gives shape and force outputs
        
    case 'point'

        if size(C{1},1)==1, C{1} = C{1}'; end % contact point in column
        CP = C{1};
        % compute fsz guess if needed
        if isempty(fsz_guess)
            fsz_guess = LOCAL_Get_PointGuess(x1,y1,z1,CP,P);
        end
        
        [~,er] = fminsearch(@LOCAL_ComputeShape_3D,[1,1,1], ...
            optimset(opt_params{:}), ...
            fsz_guess, ...
            x1,y1,z1,P,TGL_mode,C,TGL_plotsteps);
        % do same check for slip as with peg
        varargout{1} = 0; % slip flag
        if strcmp(lastwarn,'s_force > arclength(x,y,z).  Setting s_force to base of last link.')
            slip=1;
        end
        if strcmp(lastwarn,'s_force <= 0. Setting s_force to end of first link.')
            slip=1;
        end
        if er > 1e-4,
            varargout{1} = 1;
            warning('elastica3D:PointSlip','Possible point slip-off')
            slip=1;
        end % no second chance for point
        
    case 'peg'
        
        if size(C{1},1)==1, C{1} = C{1}'; end
        if size(C{2},1)==1, C{2} = C{2}'; end
        M = C{2};
        P0 = C{1};
        % compute fsz guess if needed
        if isempty(fsz_guess)
            % Find best point on peg - closest to cylinder described by peg
            % rotating around whisker base
            
            % Get peg polar distance to base
            X1 = P0; % peg origin
            X2 = P0 + M; % peg
            X0 = [x1(1);y1(1);z1(1)]; % whisker base
            d_peg = norm(cross(X0-X1,X0-X2))/norm(M);
            clear X0 X1 X2
            
            % Get whisker polar distances to base
            X0 = [x1;y1;z1]; % whisker
            L_X0 = size(X0,2);
            X1 = [x1(1);y1(1);z1(1)]; % whisker base
            X2 = X1 + M; % peg line through whisker base
            XC = cross(X0-repmat(X1,1,L_X0),X0-repmat(X2,1,L_X0));
            d_wh = sqrt(sum((XC).^2,1))./norm(M);
            clear X0 L_X0 X1 X2 XC
            
            % Get whisker node close to cylinder wall/peg
            TOL = P.s(end)/20;
            close_inds = find(abs(d_wh - d_peg)<TOL);
            switch length(close_inds)
                case 0, % nothing that close to cylinder - pick closest point
                    [~,min_ind] = min(abs(d_wh - d_peg));
                case 1, % only one point that close to cylinder
                    min_ind = close_inds;
                otherwise % pick point closest to peg
                    X0 = [x1;y1;z1];
                    X0 = X0(:,close_inds);
                    L_X0 = size(X0,2);
                    X1 = P0; % peg origin
                    X2 = P0 + M; % peg
                    XC = cross(X0-repmat(X1,1,L_X0),X0-repmat(X2,1,L_X0));
                    d_w2peg = sqrt(sum((XC).^2,1))./norm(M);
                    [~,min_min_ind] = min(d_w2peg);
                    min_ind = close_inds(min_min_ind);
                    clear L_X0 X0 X1 X2 XC
            end
            
            % Find point on peg closest to whisker "closest point"
            X0 = [x1(min_ind);y1(min_ind);z1(min_ind)];
            X1 = P0; % peg origin
            t = -dot(X1-X0,M)/(norm(M)^2);
            CP = X1 + M.*t;
            clear t X0 X1
            
            % Use this as contact point, same as 'point' mode guess
            fsz_guess = LOCAL_Get_PointGuess(x1,y1,z1,CP,P);
        end
        
        [~,er] = fminsearch(@LOCAL_ComputeShape_3D,[1,1,1],...
            optimset(opt_params{:}), ...
            fsz_guess, ...
            x1,y1,z1,P,TGL_mode,C,TGL_plotsteps);
%         disp(er)
        varargout{1} = 0; % slip flag
        if er > 1e-4
            % be able to run through rerun twice
            nrerun = 1;
            while er > 1e-4 && nrerun <= 2
                nrerun = nrerun+1;
                [~,er] = fminsearch(@LOCAL_ComputeShape_3D,[1,1,1],...
                    optimset(opt_params{:}), ...
                    LOCAL.fsz, ...
                    x1,y1,z1,P,TGL_mode,C,TGL_plotsteps);
            end
            if er > 1e-4,
                warning('elastica3D:PegSlip','Possible peg slip-off')
                varargout{1} = 1; % slip flag
                slip=1;
            end
        end
        
end

%% FINALIZE OUTPUT

% get output & move back to whisker base (not origin)
x3 = LOCAL.x2 + x1(1);
y3 = LOCAL.y2 + y1(1);
z3 = LOCAL.z2 + z1(1);

% switch force if negative
if LOCAL.fsz(1) < 0
    LOCAL.fsz(1) = -LOCAL.fsz(1);
    LOCAL.fsz(3) = LOCAL.fsz(3)+pi;
end

% group force outputs
F.fsz = LOCAL.fsz;
F.f = LOCAL.f;
F.m = LOCAL.m;
F.pos = LOCAL.pos + [x1(1);y1(1);z1(1)];

P = LOCAL.P;

% plot output
if TGL_plotfinal
%     figure
    plot3(x1,y1,z1,'k.-')
    hold on
    plot3(x3,y3,z3,'b.-');
    % make force vector 1/10 length of whisker
    f_len = P.s(end)/4;
    f_plot = [F.pos - f_len*F.f/F.fsz(1) f_len*F.f/F.fsz(1)];
    quiver3(f_plot(1,1),f_plot(2,1),f_plot(3,1),...
        f_plot(1,2),f_plot(2,2),f_plot(3,2),'r')
    plot3(F.pos(1),F.pos(2),F.pos(3),'ro')
    switch TGL_mode
        case 'point'
            plot3(CP(1),CP(2),CP(3),'go')
        case 'peg'
            z_lims = [min([z1,z3]) max([z1 z3])];
            t_min = (z_lims(1) - P0(3))/M(3);
            t_max = (z_lims(2) - P0(3))/M(3);
            Peg1 = P0 + M.*t_min;
            Peg2 = P0 + M.*t_max;
            Peg = [Peg1 Peg2];
            plot3(Peg(1,:),Peg(2,:),Peg(3,:),'g.-')
    end
    axis equal
    xlabel('x')
    ylabel('y')
    zlabel('z')
end

%% START SUBFUNCTIONS
function fsz_guess = LOCAL_Get_PointGuess(x1,y1,z1,C,P)
%% function fsz_guess = LOCAL_Get_PointGuess(x1,y1,z1,C,P)
% Computes the initial force, s_force, and zeta guess for mode 'point'
% INPUTS:
%   (x1,y1,z1) - undeflected beam shape
%   C - [x,y,z] original contact point location in 3D
%   P - struct of whisker parameters required for this function:
%       .s - arc length at every node, starting at 0
%       .E - Young's modulus at every node
%       .I - Area moment of inertia at every node
% OUTPUT:
%   fsz_guess - [force,s_force,zeta] initial guess

% Put into whisker-centered coordinates
X0 = [x1;y1;z1];
X0 = X0 - repmat([x1(1);y1(1);z1(1)],1,size(X0,2));
C = C - [x1(1);y1(1);z1(1)];
ROT = getRY(P.dphi(1))*getRZ(-P.dth(1));
X0 = ROT*X0;
C = ROT*C;
x1 = X0(1,:);
y1 = X0(2,:);
z1 = X0(3,:);

% ARC LENGTH (s) GUESS
wh_dists = sqrt(x1.^2+y1.^2+z1.^2);
obj_dist = sqrt(C(1)^2 + C(2)^2 + C(3)^2);
[~,min_ind] = min(abs(wh_dists - obj_dist));
s_guess = P.s(min_ind);

% ZETA GUESS
% put link back in its local coordinate system
if min_ind == length(x1) || wh_dists(min_ind) > obj_dist
    ind0 = min_ind-1; % base of link
else
    ind0 = min_ind;
end

link = [x1(ind0:ind0+1);y1(ind0:ind0+1);z1(ind0:ind0+1)];
C0 = C - link(:,1);
link = link - repmat(link(:,1),1,2);
th0 = atan2(link(2,2),link(1,2));
rZ = getRZ(-th0);
link = rZ*link;
C0 = rZ*C0;
phi0 = atan2(link(3,2),link(1,2));
rY = getRY(phi0);
% link = rY*link;
C0 = rY*C0;

% find contact point's angle about x-axis
z_guess = pi - atan2(C0(3),C0(2));


% FORCE GUESS
% make EI guess - use 2nd node in case of BC
ei_guess = P.E(2)*P.I(2);

displace = sqrt((x1(min_ind)-C(1))^2 + (y1(min_ind)-C(2))^2 + (z1(min_ind)-C(3))^2);

if sum(diff(P.I(2:end)))==0
    % Assume cylinder
    f_guess = 3*ei_guess*displace/(obj_dist^3);
else
    % L = P.s(end)/(1-1/P.taper); % old length => use full length for underestimate?
    L = sqrt((x1(end)-x1(1))^2 + (y1(end)-y1(1))^2 + (z1(end)-z1(1))^2);
    f_guess = 3*displace*ei_guess*(L-obj_dist)/(4*L*obj_dist^3); % Birdwell 2007 - still too big?
end

fsz_guess = [f_guess s_guess z_guess];

function er = LOCAL_ComputeShape_3D(q,FSZ_REF,x1,y1,z1,P,TGL_mode,C,TGL_plotsteps)
%% function er = LOCAL_ComputeShape_3D(fsz,x1,y1,z1,P,TGL_mode,C,TGL_plotsteps)
% Calculates new whisker position and forces and moments at base for given
% fsz input. Stores all info. to global variable, LOCAL, and outputs 'er'.
% -------------------------------------------------------------------------
% INPUTS:
%   q - a vector containing [f,s,z], Used as a multiplier for FSZ_REF. 
%       Force (f)[N] applied at arc length (s)[m] and at angle 
%       zeta (z)[radians]. Usual input: [1, 1, 1]. 
%   FSZ_REF - references for 'fsz' [f_guess s_guess z_guess]
%   [x1, y1, z1] - initial whisker shape [m]
%   P - struct of whisker parameters
%       .dth - [1x(N-1)] theta angle change between whisker nodes [radians]
%       .dphi - [1x(N-1)] phi angle change between whisker nodes [radians]
%       .ds - [1x(N-1)] length of each link [m]
%       .s - [1xN] arc length at each node, starting at 0 [m]
%       .E - [1xN] Young's modulus at each node [Pa]
%       .I - [1xN] Area moment of inertia at each node [m^4]
%       .G - [1xN] Shear modulus at each node [Pa]
%       .J - [1xN] Torsion constant at each node [m^4]
%   TGL_mode - 'force' or 'point' or 'peg'
%   C - beam bending constraint depending on the mode:
%       'force': [f,s,z] Force (f)[N] applied at arc length (s)[m] and at
%           angle zeta (z)[radians]
%       'point' {default}: [a,b,c] point location in 3D space [m]
%       'peg': {[a,b,c] [i,j,k]}  [a,b,c] defines the point location of
%           origin in 3D space for the peg [m], and [i,j,k] define the unit
%           vector describing the pegs orientation. 
%   TGL_plotsteps - 0 implies plot each iteration of the fitting algorithm
%
% OUTPUTS:
%   er - varies by mode... 
%       Mode: 'force'
%           er = 0;
%       Mode: 'point'
%           er = Euclidean distance between (a,b,c) and deflected point on
%           the beam at arclength 's'
%       Mode: 'peg'
%           er = Weighted sum of (error between contact location and peg) and (force not being normal) 

%% Initialize

global LOCAL

% Multiply scalers and references
f_mag = q(1)*FSZ_REF(1); %f_guess
s_force = q(2)*FSZ_REF(2); %s_guess
zeta = q(3)*FSZ_REF(3); %z_guess

% correct zeta wrapping
if abs(zeta)>=2*pi
    zeta = zeta - 2*pi*sign(zeta);
end
%% Setup Whisker

% Check guess for arclength (s_force) isn't longer than whisker. Set to 
% second to last element.
if s_force > P.s(end)
    s_force = P.s(end-1);
    warning('elastica3D:ShortS_Force',...
        's_force > arclength(x,y,z).  Setting s_force to base of last link.')
end

% Check guess for arclength (s_force) isn't shorter than whisker. Set to 
% second element. 
if s_force <= 0
    s_force = P.s(2);
    warning('elastica3D:LongS_Force',...
        's_force <= 0. Setting s_force to end of first link.')
end

%% Create new whisker from s_force to base

% Find guessed arclength index on original whisker
diffs = P.s - s_force;
[~,index] = min(abs(diffs));
% Grab index after s_force
if P.s(index) <= s_force, index = index + 1; end 

nNodes = index-1;
% Make sure enough nodes to perform operation
if nNodes < 3, nNodes = 3; end

% Create new whisker up to guessed arclength
s_bend = linspace(0,s_force,nNodes);
xb = interp1(P.s,x1,s_bend,'spline');
yb = interp1(P.s,y1,s_bend,'spline');
zb = interp1(P.s,z1,s_bend,'spline');

% Make new whisker the same length as original whisker
xb = [xb x1(index:end)];
yb = [yb y1(index:end)];
zb = [zb z1(index:end)];

% Break new whisker up into nodes
ds = sqrt(diff(xb).^2+diff(yb).^2+diff(zb).^2);
s = [0 cumsum(ds)];
Pb.ds = ds;
Pb.s = s;

% W is the new whisker shape (should still be the same as the original 
% whisker at this point. Initialize dth and dphi arrays. 
W = [xb;yb;zb];
dth = zeros(1,nNodes-1);
dphi = dth;

% Iterate through all nodes to get initial values for dth and dphi
for ii = 1:(length(xb)-1)
    % Pick next node at origin
    W(1,:) = W(1,:) - W(1,ii);
    W(2,:) = W(2,:) - W(2,ii);
    W(3,:) = W(3,:) - W(3,ii);
    
    % dth = atan(y/x)
    dth(ii) = atan2(W(2,ii+1),W(1,ii+1)); % z-axis rotation toward positive x axis 
    W = getRZ(-dth(ii))*W; % rotate whisker to normal alignment
    
    % dphi = atan(z/x)
    dphi(ii) = atan2(W(3,ii+1),W(1,ii+1)); % y-axis toward positive x axis 
    W = getRY(dphi(ii))*W; % rotate whisker to normal alignment
end
Pb.dth = dth;
Pb.dphi = dphi;

% Assign E, I, G, J
Pb.E(1) = P.E(1);
Pb.I(1) = P.I(1);
Pb.G(1) = P.G(1);
Pb.J(1) = P.J(1);

Pb.E(2:nNodes) = interp1(P.s(2:end),P.E(2:end),s_bend(2:end),'spline');
Pb.I(2:nNodes) = interp1(P.s(2:end),P.I(2:end),s_bend(2:end),'spline');
Pb.G(2:nNodes) = interp1(P.s(2:end),P.G(2:end),s_bend(2:end),'spline');
Pb.J(2:nNodes) = interp1(P.s(2:end),P.J(2:end),s_bend(2:end),'spline');

clear W dth dphi ds s

ds_bend = [Pb.ds(1) Pb.ds];



%% Find setup zeta
% go through entire rotation matrices
% originate index link in proper place
% first column is end of link, second is along positive y-axis
W_temp = [[Pb.ds(nNodes-1);0;0] [0;1;0] [0;0;0]];
for ii = nNodes-1:-1:2
    W_temp = getRZ(Pb.dth(ii))*getRY(-Pb.dphi(ii))*W_temp;
    W_temp(1,:) = W_temp(1,:) + Pb.ds(ii-1);
end
W_temp = W_temp - repmat(W_temp(:,3),1,3); % move link back to origin
% now have link at origin but in proper rotation matrices
% WITH RESPECT TO WHISKER BASE
% find current phi and theta angle
th_act = atan2(W_temp(2,1),W_temp(1,1));
W_temp = getRZ(-th_act)*W_temp;
phi_act = atan2(W_temp(3,1),W_temp(1,1));
W_temp = getRY(phi_act)*W_temp;
zeta_act = atan2(W_temp(3,2),W_temp(2,2));
zeta_use = zeta - zeta_act;

%% Get zeta angle
W_loc = [xb; yb; zb];

% Manipulate W_loc into position
% Move last node to origin
W_loc(1,:) = W_loc(1,:) - W_loc(1,nNodes-1);
W_loc(2,:) = W_loc(2,:) - W_loc(2,nNodes-1);
W_loc(3,:) = W_loc(3,:) - W_loc(3,nNodes-1);

% Rotate to align on x-axis
ax = cross(W_loc(:,nNodes),[1 0 0]);
n = norm(ax);
ax = ax/n;
ax(isnan(ax)) = 0;
ang = acos(dot(W_loc(:,nNodes),[1 0 0])/norm(W_loc(:,nNodes)));
W_loc = getROT(ang,ax)*W_loc;

zdes = atan2(sin(Pb.dphi(nNodes-1)),cos(Pb.dphi(nNodes-1))*sin(Pb.dth(nNodes-1)));
zact = atan2(W_loc(3,nNodes-2),W_loc(2,nNodes-2));
ang = zdes-zact;
W_loc = getRX(ang)*W_loc;

% Make final W array
W = zeros(size(W_loc));
W(:,nNodes+1:end) = W_loc(:,nNodes+1:end);

%% Get force direction
% Define force as vector
F = getRX(zeta_use)*[Pb.ds(nNodes-1) Pb.ds(nNodes-1);...
                     1               0;...
                     0               0]; % Unit vector so force is normal

%% Main loop to calculate bending
for jj = nNodes-1:-1:1
    % Get total moment
    M_cross = cross(F(:,2),f_mag*(F(:,2)-F(:,1)));  % r x F
    
    % Separate moment into bending and torque
    M_bend = [0 M_cross(2) M_cross(3)]';
    T = M_cross(1);
    
    % Implement torque
    t_ang = T*ds_bend(jj+1)/(Pb.G(jj)*Pb.J(jj)); %angle change in link's own axis
    R_T = getRX(t_ang);
    W = R_T*W;
    F = R_T*F;
    
    % Implement bending
    M = norm(M_bend);
    ax = M_bend/M;
    ax(isnan(ax))=0; % eliminate NaN if moment is zero
    ang = ds_bend(jj)*M/(Pb.E(jj)*Pb.I(jj)); %angle change in plane of bending moment
    R_M = getROT(ang,ax);
    W = R_M*W;
    F = R_M*F;
    
    % Rotate all to place next segment in oriented position
    R_node = getRZ(Pb.dth(jj))*getRY(-Pb.dphi(jj));
    W = R_node*W;
    F = R_node*F;
    
    % Translate for new segment
    W(1,jj:end) = W(1,jj:end)+ds_bend(jj);
    F(1,:) = F(1,:)+ds_bend(jj);
end

% Undo last translation for base
W(1,:) = W(1,:)-ds_bend(jj);
F(1,:) = F(1,:)-ds_bend(jj);

%% Calculate output
% Force and moment calculated with respect to whisker origin in world
% oriented frame

LOCAL.fsz = [f_mag s_force zeta];
LOCAL.f = f_mag*(F(:,2)-F(:,1));
LOCAL.m = cross(F(:,2),f_mag*(F(:,2)-F(:,1)));

LOCAL.x2 = W(1,:);
LOCAL.y2 = W(2,:);
LOCAL.z2 = W(3,:);

LOCAL.pos = F(:,2);

% get all of P together
P_all.dth  = Pb.dth;
P_all.dphi = Pb.dphi;
P_all.ds   = Pb.ds;
P_all.s    = Pb.s;
P_all.E    = [Pb.E P.E(index:end)];
P_all.I    = [Pb.I P.I(index:end)];
P_all.G    = [Pb.G P.G(index:end)];
P_all.J    = [Pb.J P.J(index:end)];

LOCAL.P = P_all;

switch TGL_mode
    case 'force'
        er = 0;
    case 'point'
        C = C{1} - [x1(1);y1(1);z1(1)]; % relative to whisker base
        er = sqrt((C(1)-F(1,2))^2+(C(2)-F(2,2))^2+(C(3)-F(3,2))^2);
    case 'peg'
        % want distance between peg and whisker to be zero AND for force to
        % be normal to peg
        M = C{2};
        CP = C{1} - [x1(1);y1(1);z1(1)]; % relative to whisker base
        X0 = F(:,2);
        X1 = CP;
        X2 = CP + M;
        er_dist = norm(cross(X0-X1,X0-X2))/norm(M);
        pct_dist = er_dist/P.s(end);
        er_norm = abs(dot(F(:,2)-F(:,1),M/norm(M)));
        pct_norm = er_norm; %abs(er_norm/LOCAL.fsz(1));
%         er = [er_dist er_norm];
        er = pct_dist + pct_norm;
end

% disp([er_dist er_norm])
% pause

% Plot
if TGL_plotsteps,
    figure(999); clf(999);
%     plot3(x1,y1,z1,'k.-')
    hold on
    plot3(W(1,:),W(2,:),W(3,:),'b.-');
    % make force vector 1/10 length of whisker
    f_len = P.s(end)/4;
    f_plot = [F(:,2) - f_len*(F(:,2)-F(:,1)) f_len*(F(:,2)-F(:,1))];
    quiver3(f_plot(1,1),f_plot(2,1),f_plot(3,1),...
        f_plot(1,2),f_plot(2,2),f_plot(3,2),'r')
    plot3(F(1,2),F(2,2),F(3,2),'ro')
    switch TGL_mode
        case 'point'
            plot3(C(1),C(2),C(3),'go')
        case 'peg'
            z_lims = [min([z1 W(3,:)]) max([z1 W(3,:)])];
            t_min = (z_lims(1) - CP(3))/M(3);
            t_max = (z_lims(2) - CP(3))/M(3);
            Peg1 = CP + M.*t_min;
            Peg2 = CP + M.*t_max;
            Peg = [Peg1 Peg2];
            plot3(Peg(1,:),Peg(2,:),Peg(3,:),'g.-')
    end
    axis equal
    xlabel('x')
    ylabel('y')
    zlabel('z')
    view(-115,50)
    disp(LOCAL.fsz)
    disp([pct_dist pct_norm])
    pause
end

function RX = getRX(theta)
%% function RX = getRX(theta)
% Create the 3x3 rotation matrix for rotation around X axis
% INPUT:  theta = angle to rotate around (radians)
% OUTPUT: RX = resulting 3x3 rotation matrix

c = cos(theta);
s = sin(theta);
RX = [1  0  0;...
      0  c -s;...
      0  s  c];
  
function RY = getRY(theta)
%% function RY = getRY(theta)
% Create 3x3 the rotation matrix for rotation around Y axis
% INPUT:  theta = angle to rotate around (radians)
% OUTPUT: RX = resulting 3x3 rotation matrix

c = cos(theta);
s = sin(theta);
RY = [c  0  s;...
      0  1  0;...
     -s  0  c];
 
function RZ = getRZ(theta)
%% function RZ = getRZ(theta)
% Create 3x3 the rotation matrix for rotation around Z axis
% INPUT:  theta = angle in radiams to rotate around
% OUTPUT: RX = resulting 3x3 rotation matrix

c = cos(theta);
s = sin(theta);
RZ = [c -s  0;...
      s  c  0;...
      0  0  1];
  
function ROT = getROT(theta,axis)
%% function ROT = getROT(theta,axis)
% create 3x3 the rotation matrix for rotation around a specific axis
% INPUTS:
%   axis  = 3D unit vector that defines the axis to be rotated around
%           If axis is non-unit vector, getROT outputs ID matrix
%   theta = angle to rotate around (radians)
% OUTPUT:
%   ROT   = resulting 3x3 rotation matrix

TOL = 1e-15;
if abs(norm(axis)-1) > TOL
    ROT = [1 0 0;...
           0 1 0;...
           0 0 1];
else
    ROT = [cos(theta)+axis(1)^2*(1-cos(theta)),...
           axis(1)*axis(2)*(1-cos(theta))-axis(3)*sin(theta),...
           axis(1)*axis(3)*(1-cos(theta))+axis(2)*sin(theta);...
           axis(1)*axis(2)*(1-cos(theta))+axis(3)*sin(theta),...
           cos(theta)+axis(2)^2*(1-cos(theta)),...
           axis(2)*axis(3)*(1-cos(theta))-axis(1)*sin(theta);...
           axis(1)*axis(3)*(1-cos(theta))-axis(2)*sin(theta),...
           axis(2)*axis(3)*(1-cos(theta))+axis(1)*sin(theta),...
           cos(theta)+axis(3)^2*(1-cos(theta))];
end