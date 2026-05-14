%% =========================================================
%  IEEE 30-Bus FAULTY Dataset Generator — COMPATIBLE VERSION
%
%  Produces exactly the SAME feature layout as the normal dataset:
%    Cols   1- 30 : |V| bus voltage magnitudes (pu)  [normalised by V_BASE]
%    Cols  31- 60 : theta voltage angles (deg)
%    Cols  61-101 : |I| branch currents (pu)          [41 branches, Ybus]
%    Cols 102-131 : P  bus active power (pu)           [Ybus]
%    Cols 132-161 : Q  bus reactive power (pu)         [Ybus]
%    Col  162     : label  1=LG  2=LL  3=LLG  4=LLL
%
%  KEY FIXES vs original:
%   1. V_BASE = 0.7420  (matches normal dataset normalisation exactly)
%   2. P, Q, I computed analytically via Ybus  (no measurement blocks needed)
%   3. Wide format output (1 row = 1 sample, 162 cols) matches normal dataset
%   4. Long format output also produced for reference
%   5. boxplot replaced — no Statistics Toolbox needed
%   6. Fault signal extraction: reads DURING-FAULT window, not end-of-sim
%   7. merge_datasets.m logic included at the end
%% =========================================================

clc; clear; close all;

%% ============================================================
%  CONFIGURATION — must match normal dataset generator exactly
%% ============================================================
MODEL_NAME        = 'fault30bussys';
FAULT_BLOCK       = 'fault30bussys/FaultBlock';
NUM_BUSES         = 30;
NUM_SAMPLES_TOTAL = 500;   % total fault samples (125 per fault type)
SAMPLES_PER_FAULT = NUM_SAMPLES_TOTAL / 4;

% *** MUST MATCH normal dataset V_BASE ***
V_BASE          = 0.7420;
TARGET_SLACK_PU = 1.05;

% Fault timing
FAULT_START = 0.1;
FAULT_END   = 0.2;
SIM_STOP    = 0.3;

% Fault resistance variation
RF_MIN = 0.001;
RF_MAX = 1.0;

% Noise — same levels as normal dataset
NOISE_V     = 0.002;
NOISE_THETA = 0.005;
NOISE_I     = 0.002;
NOISE_P     = 0.002;
NOISE_Q     = 0.002;

%% ============================================================
%  IEEE 30-BUS BRANCH DATA  (identical to normal dataset)
%% ============================================================
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
    10 21   0.0348  0.0749  0;
];
NUM_BRANCHES = size(branch_data, 1);   % 41

%% ============================================================
%  COLUMN INDEX RANGES  (identical to normal dataset)
%% ============================================================
NUM_COLS = 2*NUM_BUSES + NUM_BRANCHES + 2*NUM_BUSES + 1;  % 162
COL_V   = 1                            : NUM_BUSES;
COL_ANG = NUM_BUSES+1                  : 2*NUM_BUSES;
COL_I   = 2*NUM_BUSES+1               : 2*NUM_BUSES+NUM_BRANCHES;
COL_P   = 2*NUM_BUSES+NUM_BRANCHES+1  : 3*NUM_BUSES+NUM_BRANCHES;
COL_Q   = 3*NUM_BUSES+NUM_BRANCHES+1  : 4*NUM_BUSES+NUM_BRANCHES;
COL_LBL = NUM_COLS;

fprintf('Layout: %d buses, %d branches → %d columns\n', ...
        NUM_BUSES, NUM_BRANCHES, NUM_COLS);
fprintf('V_BASE = %.4f  (matches normal dataset)\n\n', V_BASE);

%% ============================================================
%  BUILD YBUS  (identical to normal dataset)
%% ============================================================
Ybus = zeros(NUM_BUSES);
for k = 1:NUM_BRANCHES
    fi=branch_data(k,1); ti=branch_data(k,2);
    r=branch_data(k,3);  x=branch_data(k,4);  bc=branch_data(k,5);
    if r==0 && x==0; continue; end
    ys = 1/(r+1j*x);
    Ybus(fi,fi)=Ybus(fi,fi)+ys+1j*bc;  Ybus(ti,ti)=Ybus(ti,ti)+ys+1j*bc;
    Ybus(fi,ti)=Ybus(fi,ti)-ys;        Ybus(ti,fi)=Ybus(ti,fi)-ys;
end
fprintf('Ybus built: %dx%d\n\n', size(Ybus,1), size(Ybus,2));

%% ============================================================
%  FAULT TYPE DEFINITIONS
%% ============================================================
% [PhaseA  PhaseB  PhaseC  Ground]
fault_cfg = {
    'on',  'off', 'off', 'on';    % 1 = LG
    'on',  'on',  'off', 'off';   % 2 = LL
    'on',  'on',  'off', 'on';    % 3 = LLG
    'on',  'on',  'on',  'off';   % 4 = LLL
};
fault_names  = {'LG','LL','LLG','LLL'};
fault_labels = [1, 2, 3, 4];

%% ============================================================
%  HELPER FUNCTIONS
%% ============================================================

% --- Ybus-based P, Q, branch I (same as normal dataset) ---
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
        Ibr(k)=abs((1/(r+1j*x))*(V(fi)-V(ti)));
    end
end

% --- Read one signal from simOut (all access methods) ---
function val = read_sig(simOut, nm, def)
    val = def;
    try; raw=simOut.(nm);
        if isnumeric(raw)&&~isempty(raw); val=raw(end); return; end
    catch; end
    try; e=simOut.logsout.getElement(nm); val=e.Values.Data(end); return; catch; end
    try; raw=simOut.get(nm);
        if isnumeric(raw)&&~isempty(raw); val=raw(end); return; end
        if isa(raw,'timeseries'); val=raw.Data(end); return; end
    catch; end
    try; raw=evalin('base',nm);
        if isnumeric(raw)&&~isempty(raw); val=raw(end); return; end
    catch; end
end

% --- Read a bus vector from simOut ---
function vec = read_vec(simOut, prefixes, N, def)
    vec = ones(1,N)*def;
    for b=1:N
        for p=1:length(prefixes)
            v=read_sig(simOut,sprintf('%s%d',prefixes{p},b),NaN);
            if ~isnan(v)&&isfinite(v); vec(b)=v; break; end
        end
    end
end

% --- Read DURING-FAULT window average (more informative than end value) ---
% tout and yout come from a timeseries or plain array
function val = fault_window_value(simOut, nm, t_start, t_end, def)
    val = def;
    % Try to get time vector and data
    try
        raw  = simOut.(nm);
        tout = simOut.tout;
        if isnumeric(raw) && isnumeric(tout) && length(raw)==length(tout)
            idx = tout >= t_start & tout <= t_end;
            if any(idx); val = mean(abs(raw(idx))); return; end
            val = raw(end); return;
        end
        if isnumeric(raw) && ~isempty(raw)
            val = raw(end); return;
        end
    catch; end
    % Fall back to plain end-of-sim value
    val = read_sig(simOut, nm, def);
end

% --- Read bus vector using fault-window average ---
function vec = read_vec_fault(simOut, prefixes, N, def, t_start, t_end)
    vec = ones(1,N)*def;
    for b=1:N
        for p=1:length(prefixes)
            nm = sprintf('%s%d',prefixes{p},b);
            v  = fault_window_value(simOut,nm,t_start,t_end,NaN);
            if ~isnan(v)&&isfinite(v); vec(b)=v; break; end
        end
    end
end

% --- Safe set_param ---
function safe_set(blk, param, value)
    if ~isempty(param)
        try; set_param(blk,param,value);
        catch ME; warning('set_param(%s,%s): %s',param,value,ME.message); end
    end
end

%% ============================================================
%  LOAD MODEL
%% ============================================================
fprintf('Loading model: %s\n', MODEL_NAME);
load_system(MODEL_NAME);
set_param(MODEL_NAME,'StopTime',      num2str(SIM_STOP));
set_param(MODEL_NAME,'SimulationMode','normal');
set_param(MODEL_NAME,'FastRestart',   'off');

% Verify fault block exists
try
    get_param(FAULT_BLOCK,'BlockType');
    fprintf('Fault block found: %s\n', FAULT_BLOCK);
catch
    error('Fault block not found: %s\nCheck MODEL_NAME and FAULT_BLOCK path.', FAULT_BLOCK);
end

% Configure To Workspace blocks
toWS = find_system(MODEL_NAME,'BlockType','ToWorkspace');
for k=1:length(toWS)
    try
        set_param(toWS{k},'SaveFormat','Array');
        set_param(toWS{k},'MaxDataPoints','inf');
        set_param(toWS{k},'Decimation','1');
    catch; end
end
fprintf('Configured %d To Workspace blocks\n', length(toWS));

%% ============================================================
%  AUTO-DETECT FAULT BLOCK PARAMETERS
%% ============================================================
fprintf('\nProbing fault block parameters...\n');
allP = fieldnames(get_param(FAULT_BLOCK,'ObjectParameters'));

function name = pick_param(candidates, allParams)
    name = '';
    for k=1:numel(candidates)
        if any(strcmpi(candidates{k},allParams)); name=candidates{k}; return; end
    end
end

PARAM_A   = pick_param({'FaultA','PhaseA','Faulted_A','FaultPhaseA'},        allP);
PARAM_B   = pick_param({'FaultB','PhaseB','Faulted_B','FaultPhaseB'},        allP);
PARAM_C   = pick_param({'FaultC','PhaseC','Faulted_C','FaultPhaseC'},        allP);
PARAM_GND = pick_param({'GroundFault','Ground','Faulted_G','GroundFaulted','Gnd'}, allP);
PARAM_RF  = pick_param({'FaultResistance','Rf','R_fault','RFault'},          allP);
PARAM_RG  = pick_param({'GroundResistance','Rg','R_ground','RGround'},       allP);
PARAM_SW  = pick_param({'SwitchTimes','FaultTime','Timing','Times'},         allP);

if isempty(PARAM_A)||isempty(PARAM_B)||isempty(PARAM_C)
    fprintf('\nAvailable FaultBlock parameters:\n');
    for k=1:numel(allP); fprintf('  %s\n',allP{k}); end
    error('Phase switch parameters not found. Update pick_param candidate lists above.');
end

fprintf('  Phase A=%s  B=%s  C=%s  Gnd=%s\n', PARAM_A,PARAM_B,PARAM_C,PARAM_GND);
% fprintf('  Rf=%s  Rg=%s  Timing=%s\n', PARAM_RF,PARAM_RG,PARAM_SW);

%% ============================================================
%  PRE-ALLOCATE — wide format (1 row = 1 sample, 162 cols)
%% ============================================================
dataset_fault = zeros(NUM_SAMPLES_TOTAL, NUM_COLS);
rf_log        = zeros(NUM_SAMPLES_TOTAL, 1);   % stores Rf for each sample

%% ============================================================
%  MAIN GENERATION LOOP
%% ============================================================
fprintf('\nStarting fault dataset generation (%d samples, %d per type)...\n', ...
        NUM_SAMPLES_TOTAL, SAMPLES_PER_FAULT);
tic;
global_sample = 0;
failed_count  = 0;

for f = 1:4
    fprintf('\n========================================\n');
    fprintf('Fault type %d: %s\n', f, fault_names{f});
    fprintf('========================================\n');

    % Set fault type (phase switches stay constant across all samples of this type)
    set_param(MODEL_NAME,'FastRestart','off');   % must be off to change fault type
    safe_set(FAULT_BLOCK, PARAM_A,   fault_cfg{f,1});
    safe_set(FAULT_BLOCK, PARAM_B,   fault_cfg{f,2});
    safe_set(FAULT_BLOCK, PARAM_C,   fault_cfg{f,3});
    safe_set(FAULT_BLOCK, PARAM_GND, fault_cfg{f,4});
    safe_set(FAULT_BLOCK, PARAM_SW,  sprintf('[%g %g]', FAULT_START, FAULT_END));

    for s = 1:SAMPLES_PER_FAULT
        global_sample = global_sample + 1;

        % Randomise fault resistance each sample
        Rf = RF_MIN + (RF_MAX-RF_MIN)*rand;
        Rg = Rf * 0.1;

        % FastRestart must be OFF to change Rf (structural parameter)
        set_param(MODEL_NAME,'FastRestart','off');
        safe_set(FAULT_BLOCK, PARAM_RF, num2str(Rf));
        if ~isempty(PARAM_RG)
            safe_set(FAULT_BLOCK, PARAM_RG, num2str(Rg));
        end

        % Run simulation
        try
            simOut = sim(MODEL_NAME,'ReturnWorkspaceOutputs','on');
        catch ME
            warning('Sample %d (%s) failed: %s', global_sample, fault_names{f}, ME.message);
            failed_count = failed_count+1;
            continue;
        end

        % Extract voltage magnitudes and angles
        % Use fault-window average for more fault-informative features
        V_raw = read_vec_fault(simOut, {'v','V','Vm','Vbus'}, NUM_BUSES, ...
                               v1_nom_fallback(V_BASE, TARGET_SLACK_PU), ...
                               FAULT_START, FAULT_END);
        A_raw = read_vec_fault(simOut, {'ang','theta','Va','angle'}, NUM_BUSES, ...
                               0.0, FAULT_START, FAULT_END);

        % Normalise to pu using SAME V_BASE as normal dataset
        V_pu = V_raw / V_BASE;

        % Compute P, Q, I from Ybus (same method as normal dataset)
        [P_bus, Q_bus, I_br] = compute_pqi(V_pu, A_raw, Ybus, branch_data);

        % Add measurement noise (same levels as normal dataset)
        V_pu  = V_pu  + NOISE_V     * randn(1, NUM_BUSES);
        A_raw = A_raw + NOISE_THETA * randn(1, NUM_BUSES);
        I_br  = I_br  + NOISE_I     * randn(1, NUM_BRANCHES);
        P_bus = P_bus + NOISE_P     * randn(1, NUM_BUSES);
        Q_bus = Q_bus + NOISE_Q     * randn(1, NUM_BUSES);

        % Store into pre-allocated matrix
        dataset_fault(global_sample, COL_V)   = V_pu;
        dataset_fault(global_sample, COL_ANG) = A_raw;
        dataset_fault(global_sample, COL_I)   = I_br;
        dataset_fault(global_sample, COL_P)   = P_bus;
        dataset_fault(global_sample, COL_Q)   = Q_bus;
        dataset_fault(global_sample, COL_LBL) = fault_labels(f);
        rf_log(global_sample)                 = Rf;   % record exact Rf used

        if mod(s,50)==0 || s==1
            elapsed=toc;
            done=(f-1)*SAMPLES_PER_FAULT+s;
            total=NUM_SAMPLES_TOTAL;
            fprintf('  [%s] %3d/%d | Total %4d/%d | ETA %.0fs\n', ...
                fault_names{f},s,SAMPLES_PER_FAULT,done,total,(elapsed/done)*(total-done));
        end
    end
end

set_param(MODEL_NAME,'FastRestart','off');

%% ============================================================
%  HELPER (needed above — define before use in older MATLAB)
%% ============================================================
function v = v1_nom_fallback(V_BASE, TARGET)
    v = V_BASE * TARGET;
end

%% ============================================================
%  FILTER INVALID ROWS
%% ============================================================
V_check = dataset_fault(:, COL_V);
valid   = any(dataset_fault~=0, 2) & all(V_check>=0, 2);
dataset_fault_clean = dataset_fault(valid, :);
rf_log_clean        = rf_log(valid);           % keep rf_log in sync with cleaned rows
fprintf('\nValid fault samples: %d / %d   Failed: %d\n', ...
        sum(valid), NUM_SAMPLES_TOTAL, failed_count);

%% ============================================================
%  SANITY CHECK
%% ============================================================
fprintf('\n--- Sanity Check ---\n');
fprintf('|V|  : [%.4f, %.4f] pu    (normal was [0.90,1.10]; fault should be lower)\n', ...
        min(min(dataset_fault_clean(:,COL_V))), ...
        max(max(dataset_fault_clean(:,COL_V))));
fprintf('ang  : [%.2f, %.2f] deg\n', ...
        min(min(dataset_fault_clean(:,COL_ANG))), ...
        max(max(dataset_fault_clean(:,COL_ANG))));
fprintf('|I|  : [%.4f, %.4f] pu    (should be >> normal during fault)\n', ...
        min(min(dataset_fault_clean(:,COL_I))), ...
        max(max(dataset_fault_clean(:,COL_I))));
fprintf('P    : [%.4f, %.4f] pu\n', ...
        min(min(dataset_fault_clean(:,COL_P))), ...
        max(max(dataset_fault_clean(:,COL_P))));
fprintf('Q    : [%.4f, %.4f] pu\n', ...
        min(min(dataset_fault_clean(:,COL_Q))), ...
        max(max(dataset_fault_clean(:,COL_Q))));

fprintf('\nSample counts per fault type:\n');
for f=1:4
    n = sum(dataset_fault_clean(:,COL_LBL)==fault_labels(f));
    fprintf('  %s (label=%d): %d samples\n', fault_names{f}, fault_labels(f), n);
end

%% ============================================================
%  SAVE — WIDE FORMAT (matches normal_dataset.csv structure)
%% ============================================================
save('fault_dataset.mat','dataset_fault_clean','rf_log_clean', ...
     'NUM_BUSES','NUM_BRANCHES','COL_V','COL_ANG','COL_I','COL_P','COL_Q','V_BASE');

col_names = [arrayfun(@(b)sprintf('V_mag_bus%d',b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('V_ang_bus%d',b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('I_mag_br%d', b), 1:NUM_BRANCHES, 'UniformOutput',false), ...
             arrayfun(@(b)sprintf('P_bus%d',    b), 1:NUM_BUSES,    'UniformOutput',false), ...
             arrayfun(@(b)sprintf('Q_bus%d',    b), 1:NUM_BUSES,    'UniformOutput',false), ...
             {'label'}];
T_fault = array2table(dataset_fault_clean, 'VariableNames', col_names);
T_fault.fault_resistance = rf_log_clean;       % append rf as last column
writetable(T_fault, 'fault_dataset.csv');
fprintf('\nSaved fault_dataset.csv  (%d x %d)\n', ...
        size(dataset_fault_clean,1), size(dataset_fault_clean,2));

%% ============================================================
%  SAVE — LONG FORMAT (per-bus rows, matches normal_dataset_long.csv)
%% ============================================================
nf   = size(dataset_fault_clean,1);
rows = nf * NUM_BUSES;
sid=zeros(rows,1); bid=zeros(rows,1);
vm=zeros(rows,1);  va=zeros(rows,1);
im=zeros(rows,1);  pp=zeros(rows,1);
qq=zeros(rows,1);  ll=zeros(rows,1);
r=1;
for s=1:nf
    for b=1:NUM_BUSES
        sid(r)=s; bid(r)=b;
        vm(r)=dataset_fault_clean(s,b);
        va(r)=dataset_fault_clean(s,NUM_BUSES+b);
        if b<=NUM_BRANCHES; im(r)=dataset_fault_clean(s,2*NUM_BUSES+b); end
        pp(r)=dataset_fault_clean(s,COL_P(1)+b-1);
        qq(r)=dataset_fault_clean(s,COL_Q(1)+b-1);
        ll(r)=dataset_fault_clean(s,COL_LBL);
        r=r+1;
    end
end
T_fault_long = table(sid,bid,vm,va,im,pp,qq,ll, ...
    'VariableNames',{'sample_id','bus_id','V_mag','V_ang','I_mag','P','Q','label'});
writetable(T_fault_long,'fault_dataset_long.csv');
save('fault_dataset_long.mat','T_fault_long');

%% ============================================================
%  MERGE WITH NORMAL DATASET → full_dataset for ML training
%  Labels: 0=Normal  1=LG  2=LL  3=LLG  4=LLL
%% ============================================================
fprintf('\n--- Merging with normal dataset ---\n');

if exist('normal_dataset.mat','file')
    load('normal_dataset.mat','dataset_clean');

    % Verify column count matches
    if size(dataset_clean,2) ~= size(dataset_fault_clean,2)
        error(['Column mismatch!\n' ...
               '  normal_dataset:  %d cols\n' ...
               '  fault_dataset:   %d cols\n' ...
               'Regenerate both datasets with the same scripts.'], ...
               size(dataset_clean,2), size(dataset_fault_clean,2));
    end

    full_dataset = [dataset_clean; dataset_fault_clean];

    % Reassign sequential sample IDs in long-format merge
    fprintf('Normal samples : %d\n', size(dataset_clean,1));
    fprintf('Fault  samples : %d\n', size(dataset_fault_clean,1));
    fprintf('Total  samples : %d\n', size(full_dataset,1));
    fprintf('Columns        : %d  (162 = 30+30+41+30+30+1)\n', size(full_dataset,2));

    % Class balance check
    for lbl=0:4
        n=sum(full_dataset(:,COL_LBL)==lbl);
        names={'Normal','LG','LL','LLG','LLL'};
        fprintf('  Class %d %-8s: %4d samples (%.1f%%)\n', ...
                lbl, names{lbl+1}, n, 100*n/size(full_dataset,1));
    end

    % Save wide format
    save('full_dataset.mat','full_dataset','col_names', ...
         'NUM_BUSES','NUM_BRANCHES','COL_V','COL_ANG','COL_I','COL_P','COL_Q','V_BASE');
    T_full = array2table(full_dataset, 'VariableNames', col_names);
    writetable(T_full, 'full_dataset.csv');
    fprintf('\nSaved full_dataset.csv  (%d x %d)\n', ...
            size(full_dataset,1), size(full_dataset,2));
else
    fprintf('normal_dataset.mat not found — skipping merge.\n');
    fprintf('Run generate_normal_dataset_v3.m first, then re-run this script.\n');
end

%% ============================================================
%  VISUALISE  (no Statistics Toolbox needed)
%% ============================================================
figure('Name','Fault Dataset Overview','Color','w','NumberTitle','off');

subplot(2,3,1);
hold on;
colors = lines(4);
for f=1:4
    idx = dataset_fault_clean(:,COL_LBL)==f;
    dat = dataset_fault_clean(idx, COL_V);
    mn  = mean(dat,1);
    plot(1:NUM_BUSES, mn, 'Color',colors(f,:),'LineWidth',1.5,'DisplayName',fault_names{f});
end
title('Mean |V| per fault type (pu)'); xlabel('Bus'); ylabel('|V| (pu)');
legend; grid on; hold off;

subplot(2,3,2);
hold on;
for f=1:4
    idx = dataset_fault_clean(:,COL_LBL)==f;
    dat = dataset_fault_clean(idx, COL_I);
    mn  = mean(dat,1);
    plot(1:NUM_BRANCHES, mn,'Color',colors(f,:),'LineWidth',1.5,'DisplayName',fault_names{f});
end
title('Mean |I| per fault type (pu)'); xlabel('Branch'); ylabel('|I| (pu)');
legend; grid on; hold off;

subplot(2,3,3);
hold on;
for f=1:4
    idx  = dataset_fault_clean(:,COL_LBL)==f;
    Vbus1= dataset_fault_clean(idx,1);
    histogram(Vbus1,20,'FaceColor',colors(f,:),'FaceAlpha',0.5,'DisplayName',fault_names{f});
end
title('Bus 1 |V| by fault type'); xlabel('|V| (pu)'); ylabel('Count');
legend; grid on; hold off;

subplot(2,3,4);
hold on;
for f=1:4
    idx = dataset_fault_clean(:,COL_LBL)==f;
    dat = dataset_fault_clean(idx, COL_P);
    mn  = mean(dat,1);
    plot(1:NUM_BUSES,mn,'Color',colors(f,:),'LineWidth',1.5,'DisplayName',fault_names{f});
end
title('Mean P per fault type (pu)'); xlabel('Bus'); ylabel('P (pu)');
legend; grid on; hold off;

subplot(2,3,5);
hold on;
for f=1:4
    idx = dataset_fault_clean(:,COL_LBL)==f;
    dat = dataset_fault_clean(idx, COL_Q);
    mn  = mean(dat,1);
    plot(1:NUM_BUSES,mn,'Color',colors(f,:),'LineWidth',1.5,'DisplayName',fault_names{f});
end
title('Mean Q per fault type (pu)'); xlabel('Bus'); ylabel('Q (pu)');
legend; grid on; hold off;

subplot(2,3,6);
lbl_vals = dataset_fault_clean(:,COL_LBL);
histogram(lbl_vals,4,'FaceColor',[0 0.45 0.74],'EdgeColor','w');
xticks(1:4); xticklabels(fault_names);
title('Sample count per fault class'); xlabel('Fault type'); ylabel('Count');
grid on;

sgtitle(sprintf('IEEE 30-Bus Fault Dataset  (n=%d)', size(dataset_fault_clean,1)));
saveas(gcf,'fault_dataset_overview.png');

fprintf('\n✅  Done!\n');
fprintf('   fault_dataset.csv          (%d x %d)\n', ...
        size(dataset_fault_clean,1), NUM_COLS);
fprintf('   fault_dataset_long.csv     (%d rows x 8 cols)\n', rows);
if exist('full_dataset.csv','file')
    fprintf('   full_dataset.csv           (normal + fault, ready for ML)\n');
end
fprintf('   fault_dataset_overview.png\n');
fprintf('Total time: %.1f s\n', toc);