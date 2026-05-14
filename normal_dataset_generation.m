%% =========================================================
%  IEEE 30-Bus Normal Operating Dataset — FINAL v3
%
%  Column layout (162 total):
%    Cols   1- 30 : |V| bus voltage magnitudes (pu)
%    Cols  31- 60 : theta bus voltage angles (deg)
%    Cols  61-101 : |I| branch current magnitudes (pu)  [41 branches]
%    Cols 102-131 : P  bus active power injections (pu)
%    Cols 132-161 : Q  bus reactive power injections (pu)
%    Col  162     : label = 0 (Normal)
%% =========================================================

clc; clear; close all;

%% ---- CONFIGURATION ----
MODEL_NAME   = 'Thirtybussys';
NUM_SAMPLES  = 500;
NUM_BUSES    = 30;

LOAD_MIN = 0.85;  LOAD_MAX = 1.15;
GEN_V_MIN = 0.95; GEN_V_MAX = 1.05;

NOISE_V     = 0.002;
NOISE_THETA = 0.005;
NOISE_I     = 0.002;
NOISE_P     = 0.002;
NOISE_Q     = 0.002;

TARGET_SLACK_PU = 1.05;   % intended slack bus pu (standard IEEE 30-bus)

%% ---- IEEE 30-BUS BRANCH DATA ----
% [from  to  R(pu)  X(pu)  B/2(pu)]  on 100 MVA base
branch_data = [
     1  2   0.0192  0.0575  0.0264;
     1  3   0.0452  0.1852  0.0204;
     2  4   0.0570  0.1737  0.0184;
     3  4   0.0132  0.0379  0.0042;
     2  5   0.0472  0.1983  0.0209;
     2  6   0.0581  0.1763  0.0187;
     4  6   0.0119  0.0414  0.0045;
     5  7   0.0460  0.1160  0.0102;
     6  7   0.0267  0.0820  0.0085;
     6  8   0.0120  0.0420  0.0045;
     6  9   0       0.2080  0;
     6 10   0       0.5560  0;
     9 11   0       0.2080  0;
     9 10   0       0.1100  0;
    12 13   0       0.1400  0;
    12 14   0.1231  0.2559  0;
    12 15   0.0662  0.1304  0;
    12 16   0.0945  0.1987  0;
    14 15   0.2210  0.1997  0;
    16 17   0.0824  0.1923  0;
    15 18   0.1070  0.2185  0;
    18 19   0.0639  0.1292  0;
    19 20   0.0340  0.0680  0;
    10 20   0.0936  0.2090  0;
    10 17   0.0324  0.0845  0;
    10 21   0.0348  0.0749  0;
    10 22   0.0727  0.1499  0;
    21 22   0.0116  0.0236  0;
    15 23   0.1000  0.2020  0;
    22 24   0.1150  0.1790  0;
    23 24   0.1320  0.2700  0;
    24 25   0.1885  0.3292  0;
    25 26   0.2544  0.3800  0;
    25 27   0.1093  0.2087  0;
    28 27   0       0.3960  0;
     6 28   0.0169  0.0599  0.0065;
     8 28   0.0636  0.2000  0.0214;
    27 29   0.2198  0.4153  0;
    27 30   0.3202  0.6027  0;
    29 30   0.2399  0.4533  0;
    10 21   0.0348  0.0749  0;   % branch 41 (parallel)
];
NUM_BRANCHES = size(branch_data, 1);   % 41

% Total columns = 2*NUM_BUSES + NUM_BRANCHES + 2*NUM_BUSES + 1
NUM_COLS = 2*NUM_BUSES + NUM_BRANCHES + 2*NUM_BUSES + 1;
fprintf('Layout: %d buses, %d branches → %d columns per sample\n', ...
        NUM_BUSES, NUM_BRANCHES, NUM_COLS);

% Column index helpers (used throughout)
COL_V   = 1               : NUM_BUSES;
COL_ANG = NUM_BUSES+1     : 2*NUM_BUSES;
COL_I   = 2*NUM_BUSES+1   : 2*NUM_BUSES+NUM_BRANCHES;
COL_P   = 2*NUM_BUSES+NUM_BRANCHES+1   : 3*NUM_BUSES+NUM_BRANCHES;
COL_Q   = 3*NUM_BUSES+NUM_BRANCHES+1   : 4*NUM_BUSES+NUM_BRANCHES;
COL_LBL = NUM_COLS;

%% ---- BUILD YBUS ----
Ybus = zeros(NUM_BUSES);
for k = 1:NUM_BRANCHES
    fi = branch_data(k,1);  ti = branch_data(k,2);
    r  = branch_data(k,3);  x  = branch_data(k,4);  bc = branch_data(k,5);
    if r==0 && x==0; continue; end
    ys = 1/(r+1j*x);
    Ybus(fi,fi) = Ybus(fi,fi) + ys + 1j*bc;
    Ybus(ti,ti) = Ybus(ti,ti) + ys + 1j*bc;
    Ybus(fi,ti) = Ybus(fi,ti) - ys;
    Ybus(ti,fi) = Ybus(ti,fi) - ys;
end
fprintf('Ybus: %dx%d\n', size(Ybus,1), size(Ybus,2));

%% ---- HELPER FUNCTIONS ----

function [P,Q,Ibr] = compute_pqi(Vmag, Vdeg, Ybus, bdata)
    V   = Vmag(:) .* exp(1j*deg2rad(Vdeg(:)));
    S   = V .* conj(Ybus * V);
    P   = real(S)';
    Q   = imag(S)';
    nb  = size(bdata,1);
    Ibr = zeros(1,nb);
    for k = 1:nb
        fi=bdata(k,1); ti=bdata(k,2);
        r=bdata(k,3);  x=bdata(k,4);
        if r==0 && x==0; continue; end
        Ibr(k) = abs((1/(r+1j*x))*(V(fi)-V(ti)));
    end
end

function val = read_signal(simOut, vname, default_val)
    val = default_val;
    % Method 1: direct field (SaveFormat=Array)
    try
        raw = simOut.(vname);
        if isnumeric(raw) && ~isempty(raw)
            val = raw(end); return;
        end
    catch; end
    % Method 2: logsout
    try
        e = simOut.logsout.getElement(vname);
        val = e.Values.Data(end); return;
    catch; end
    % Method 3: get()
    try
        raw = simOut.get(vname);
        if isnumeric(raw) && ~isempty(raw)
            val = raw(end); return;
        end
        if isa(raw,'timeseries')
            val = raw.Data(end); return;
        end
    catch; end
    % Method 4: base workspace fallback
    try
        raw = evalin('base', vname);
        if isnumeric(raw) && ~isempty(raw)
            val = raw(end); return;
        end
    catch; end
end

function vec = read_bus_vector(simOut, prefixes, N, default_val)
    vec = ones(1,N)*default_val;
    for b = 1:N
        for p = 1:length(prefixes)
            v = read_signal(simOut, sprintf('%s%d',prefixes{p},b), NaN);
            if ~isnan(v) && isfinite(v)
                vec(b) = v; break;
            end
        end
    end
end

function ok = try_set(blk, param, val)
    ok = false;
    try; set_param(blk, param, val); ok = true; catch; end
end

%% ---- LOAD MODEL ----
fprintf('\nLoading model: %s\n', MODEL_NAME);
load_system(MODEL_NAME);
set_param(MODEL_NAME,'FastRestart','off');
set_param(MODEL_NAME,'SimulationMode','normal');

% Configure all To Workspace blocks → Array format
toWS = find_system(MODEL_NAME,'BlockType','ToWorkspace');
for k = 1:length(toWS)
    try
        set_param(toWS{k},'SaveFormat','Array');
        set_param(toWS{k},'MaxDataPoints','inf');
        set_param(toWS{k},'Decimation','1');
    catch; end
end
fprintf('Configured %d To Workspace blocks\n', length(toWS));

% Discover generator/load blocks
gen_blocks  = find_system(MODEL_NAME,'RegExp','on','Name','Gen\d*');
if isempty(gen_blocks)
    gen_blocks = find_system(MODEL_NAME,'RegExp','on','Name','SM\d*');
end
load_blocks = find_system(MODEL_NAME,'RegExp','on','Name','[Ll]oad\d*');
fprintf('Found %d generator blocks, %d load blocks\n', ...
        length(gen_blocks), length(load_blocks));

GEN_V_PARAMS  = {'Vf','V','Vt','VoltageSetpoint','Uref','ExcitationVoltage'};
LOAD_P_PARAMS = {'ActivePower','P','Pref','NominalPower','Pactive'};
LOAD_Q_PARAMS = {'ReactivePower','Q','Qref','Qreactive'};

%% ---- NOMINAL RUN ----
fprintf('\nRunning nominal simulation...\n');
simOut_nom = sim(MODEL_NAME,'ReturnWorkspaceOutputs','on');

v1_nom = read_signal(simOut_nom,'v1',NaN);
fprintf('Nominal v1 = %.4f\n', v1_nom);
if isnan(v1_nom) || v1_nom == 0
    warning('v1 not found — check To Workspace block for bus 1 is connected.');
    V_BASE = 1.0;
else
    V_BASE = v1_nom / TARGET_SLACK_PU;
    fprintf('V_BASE = %.4f  (Bus 1 will read %.2f pu after normalisation)\n', ...
            V_BASE, TARGET_SLACK_PU);
end

% Store nominal loads
nom_P = zeros(1,length(load_blocks));
nom_Q = zeros(1,length(load_blocks));
for l = 1:length(load_blocks)
    for pn = LOAD_P_PARAMS
        try
            v = str2double(get_param(load_blocks{l},pn{1}));
            if ~isnan(v) && v~=0; nom_P(l)=v; break; end
        catch; end
    end
    for pn = LOAD_Q_PARAMS
        try
            v = str2double(get_param(load_blocks{l},pn{1}));
            if ~isnan(v) && v~=0; nom_Q(l)=v; break; end
        catch; end
    end
end
fprintf('Nominal loads stored: %d P values, %d Q values\n', ...
        sum(nom_P~=0), sum(nom_Q~=0));

% Warn if Q loads not found (will still work — P variation alone is enough)
if sum(nom_Q~=0) == 0
    fprintf(['NOTE: No Q load parameters found by name. Only P will be varied.\n' ...
             'Q values will be computed analytically from Ybus.\n']);
end

%% ---- PRE-ALLOCATE with correct column count ----
dataset = zeros(NUM_SAMPLES, NUM_COLS);  % NUM_COLS = 162 for 30 buses, 41 branches

%% ---- ENABLE FAST RESTART ----
set_param(MODEL_NAME,'FastRestart','on');

%% ---- MAIN LOOP ----
fprintf('\nGenerating %d samples...\n', NUM_SAMPLES);
tic;
failed_count = 0;

for i = 1:NUM_SAMPLES

    % Vary generator voltage setpoints
    for g = 1:length(gen_blocks)
        Vg = GEN_V_MIN + (GEN_V_MAX-GEN_V_MIN)*rand;
        for pn = GEN_V_PARAMS
            if try_set(gen_blocks{g}, pn{1}, num2str(Vg)); break; end
        end
    end

    % Vary loads from stored nominal (prevents drift)
    scale = LOAD_MIN + (LOAD_MAX-LOAD_MIN)*rand;  % one scale per sample
    for l = 1:length(load_blocks)
        if nom_P(l)~=0
            for pn = LOAD_P_PARAMS
                if try_set(load_blocks{l},pn{1},num2str(nom_P(l)*scale)); break; end
            end
        end
        if nom_Q(l)~=0
            for pn = LOAD_Q_PARAMS
                if try_set(load_blocks{l},pn{1},num2str(nom_Q(l)*scale)); break; end
            end
        end
    end

    % Run simulation
    try
        simOut = sim(MODEL_NAME,'ReturnWorkspaceOutputs','on');
    catch ME
        warning('Sample %d failed: %s', i, ME.message);
        failed_count = failed_count+1;
        continue;
    end

    % Extract voltages and angles
    V_raw = read_bus_vector(simOut, {'v','V','Vm','Vbus'}, NUM_BUSES, v1_nom);
    A_raw = read_bus_vector(simOut, {'ang','theta','Va','angle'}, NUM_BUSES, 0.0);

    % Normalise voltages to pu
    V_pu = V_raw / V_BASE;

    % Compute P, Q, I analytically from Ybus
    [P_bus, Q_bus, I_br] = compute_pqi(V_pu, A_raw, Ybus, branch_data);

    % Add measurement noise
    V_pu  = V_pu  + NOISE_V     * randn(1, NUM_BUSES);
    A_raw = A_raw + NOISE_THETA * randn(1, NUM_BUSES);
    I_br  = I_br  + NOISE_I     * randn(1, NUM_BRANCHES);
    P_bus = P_bus + NOISE_P     * randn(1, NUM_BUSES);
    Q_bus = Q_bus + NOISE_Q     * randn(1, NUM_BUSES);

    % Store — columns exactly match COL_* index ranges
    dataset(i, COL_V)   = V_pu;
    dataset(i, COL_ANG) = A_raw;
    dataset(i, COL_I)   = I_br;
    dataset(i, COL_P)   = P_bus;
    dataset(i, COL_Q)   = Q_bus;
    dataset(i, COL_LBL) = 0;

    if mod(i,100)==0 || i==1
        elapsed = toc;
        fprintf('Sample %4d/%d | %.1fs | ETA %.1fs\n', ...
                i, NUM_SAMPLES, elapsed, (elapsed/i)*(NUM_SAMPLES-i));
    end
end

set_param(MODEL_NAME,'FastRestart','off');

%% ---- FILTER INVALID ROWS ----
V_check       = dataset(:, COL_V);
valid         = all(V_check > 0.5, 2) & all(V_check < 1.5, 2) & any(dataset~=0,2);
dataset_clean = dataset(valid,:);
fprintf('\nValid: %d / %d   Failed: %d\n', sum(valid), NUM_SAMPLES, failed_count);

%% ---- SANITY CHECK ----
fprintf('\n--- Sanity Check ---\n');
fprintf('|V|  : [%.4f, %.4f] pu     expect [0.90, 1.10]\n', ...
        min(min(dataset_clean(:,COL_V))),   max(max(dataset_clean(:,COL_V))));
fprintf('ang  : [%.2f, %.2f] deg   expect [-30, +30]\n', ...
        min(min(dataset_clean(:,COL_ANG))), max(max(dataset_clean(:,COL_ANG))));
fprintf('|I|  : [%.4f, %.4f] pu     expect [0, 1.5]\n', ...
        min(min(dataset_clean(:,COL_I))),   max(max(dataset_clean(:,COL_I))));
fprintf('P    : [%.4f, %.4f] pu     expect [-3, +3]\n', ...
        min(min(dataset_clean(:,COL_P))),   max(max(dataset_clean(:,COL_P))));
fprintf('Q    : [%.4f, %.4f] pu     expect [-1, +1]\n', ...
        min(min(dataset_clean(:,COL_Q))),   max(max(dataset_clean(:,COL_Q))));

%% ---- SAVE ----
save('normal_dataset.mat','dataset_clean','NUM_BUSES','NUM_BRANCHES','COL_V','COL_ANG','COL_I','COL_P','COL_Q');

col_names = [arrayfun(@(b)sprintf('V_mag_bus%d',b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('V_ang_bus%d',b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('I_mag_br%d', b), 1:NUM_BRANCHES, 'UniformOutput',false), ...
             arrayfun(@(b)sprintf('P_bus%d',    b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('Q_bus%d',    b), 1:NUM_BUSES,    'UniformOutput',false), ...
             {'label'}];
T = array2table(dataset_clean, 'VariableNames', col_names);
writetable(T, 'normal_dataset.csv');

%% ---- LONG FORMAT ----
nv = size(dataset_clean,1);
rows = nv*NUM_BUSES;
sid=zeros(rows,1); bid=zeros(rows,1);
vm=zeros(rows,1); va=zeros(rows,1);
im=zeros(rows,1); pp=zeros(rows,1);
qq=zeros(rows,1); ll=zeros(rows,1);
r=1;
for s=1:nv
    for b=1:NUM_BUSES
        sid(r)=s; bid(r)=b;
        vm(r)=dataset_clean(s,b);
        va(r)=dataset_clean(s,NUM_BUSES+b);
        if b<=NUM_BRANCHES; im(r)=dataset_clean(s,2*NUM_BUSES+b); end
        pp(r)=dataset_clean(s,COL_P(1)+b-1);
        qq(r)=dataset_clean(s,COL_Q(1)+b-1);
        r=r+1;
    end
end
T_long=table(sid,bid,vm,va,im,pp,qq,ll,...
    'VariableNames',{'sample_id','bus_id','V_mag','V_ang','I_mag','P','Q','label'});
writetable(T_long,'normal_dataset_long.csv');
save('normal_dataset_long.mat','T_long');

%% ---- PLOT ----
figure('Name','Normal Dataset','Color','w');
nplot = min(size(dataset_clean,1),200);

subplot(2,3,1);
hold on;
for s=1:nplot
    plot(1:NUM_BUSES,dataset_clean(s,COL_V),'Color',[0 0.45 0.74 0.12],'LineWidth',0.5);
end
plot(1:NUM_BUSES,mean(dataset_clean(:,COL_V)),'r-','LineWidth',2);
title('|V| (pu)'); xlabel('Bus'); grid on; hold off;

subplot(2,3,2);
hold on;
for s=1:nplot
    plot(1:NUM_BUSES,dataset_clean(s,COL_ANG),'Color',[0.85 0.33 0.10 0.12],'LineWidth',0.5);
end
plot(1:NUM_BUSES,mean(dataset_clean(:,COL_ANG)),'b-','LineWidth',2);
title('\theta (deg)'); xlabel('Bus'); grid on; hold off;

subplot(2,3,3);
hold on;
for s=1:nplot
    plot(1:NUM_BRANCHES,dataset_clean(s,COL_I),'Color',[0.49 0.18 0.56 0.12],'LineWidth',0.5);
end
plot(1:NUM_BRANCHES,mean(dataset_clean(:,COL_I)),'r-','LineWidth',2);
title('|I| branch (pu)'); xlabel('Branch'); grid on; hold off;

subplot(2,3,4);
hold on;
for s=1:nplot
    plot(1:NUM_BUSES,dataset_clean(s,COL_P),'Color',[0.47 0.67 0.19 0.12],'LineWidth',0.5);
end
plot(1:NUM_BUSES,mean(dataset_clean(:,COL_P)),'r-','LineWidth',2);
title('P (pu)'); xlabel('Bus'); grid on; hold off;

subplot(2,3,5);
hold on;
for s=1:nplot
    plot(1:NUM_BUSES,dataset_clean(s,COL_Q),'Color',[0.30 0.30 0.30 0.12],'LineWidth',0.5);
end
plot(1:NUM_BUSES,mean(dataset_clean(:,COL_Q)),'r-','LineWidth',2);
title('Q (pu)'); xlabel('Bus'); grid on; hold off;

subplot(2,3,6);
histogram(dataset_clean(:,1),25,'FaceColor',[0 0.45 0.74],'EdgeColor','w');
xline(mean(dataset_clean(:,1)),'r-','LineWidth',2);
title('Bus 1 |V| distribution'); xlabel('|V| (pu)'); ylabel('Count'); grid on;

sgtitle(sprintf('IEEE 30-Bus Normal Operating Dataset  (n=%d)',size(dataset_clean,1)));
saveas(gcf,'normal_dataset_overview.png');

fprintf('\n✅  Done!\n');
fprintf('   normal_dataset.csv         (%d samples × %d cols)\n', ...
        size(dataset_clean,1), size(dataset_clean,2));
fprintf('   normal_dataset_long.csv    (%d rows × 8 cols)\n', rows);
fprintf('   normal_dataset_overview.png\n');
fprintf('Total time: %.1f s\n', toc);