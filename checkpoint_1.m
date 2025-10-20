function checkpoint_1()
% MATLAB port of src/milestone.py
%
% This script/function reproduces the functionality of the Python version:
% - Loading CST element patterns from a directory
% - Computing initial tapered, steered weights
% - Running gradient-based optimization to match a target pattern
% - Evaluating grid of pointings and saving ΔNMSE plots
%
% Entry point runs the same default evaluation grid as the Python file.
    tic()
    env = 'Env1_rotated';
    taper = 'hamming';
    loss_scale = 'linear';
    lr = 1e-6;
    evaluate_grid('no_env_rotated', env, taper, @mse_loss, loss_scale, lr);
    fprintf('total runtime is: %.0f sec = %.1f min', toc(), toc()/60);
end

% ========================= Top-level API =========================
function evaluate_grid(env0_name, env1_name, taper, loss_fn, loss_scale, lr)
    if nargin < 1 || isempty(env0_name), env0_name = 'no_env_rotated'; end
    if nargin < 2 || isempty(env1_name), env1_name = 'Env1_rotated'; end
    if nargin < 3 || isempty(taper),     taper = 'hamming'; end
    if nargin < 4 || isempty(loss_fn),   loss_fn = @mse_loss; end
    if nargin < 5 || isempty(loss_scale), loss_scale = 'db'; end
    if nargin < 6 || isempty(lr),        lr = 5e-6; end

    elevs = 0:15:30;      % 0, 15, 30
    azims = 0:45:315;     % 0, 90, 180, 270

    % Evaluate optimization at each grid point
    results = cell(numel(elevs), numel(azims));
    for ei = 1:numel(elevs)
        for ai = 1:numel(azims)
            elev_deg = elevs(ei);
            azim_deg = azims(ai);
            res = run_optimization(env0_name, env1_name, taper, elev_deg, azim_deg, loss_fn, loss_scale, lr, false);
            results{ei, ai} = res;
        end
    end

    % Aggregate NMSE
    nmses_env1 = zeros(numel(elevs), numel(azims));
    nmses_opt  = zeros(numel(elevs), numel(azims));
    for ei = 1:numel(elevs)
        for ai = 1:numel(azims)
            r = results{ei, ai};
            nmses_env1(ei, ai) = r.nmse_env1;
            nmses_opt(ei, ai)  = r.nmse_opt;
        end
    end

    nmse_diff = nmses_opt - nmses_env1;
    fprintf('Mean NMSE Improvement: %.3f dB\n', mean(nmse_diff(:)));
    fprintf('Std NMSE Improvement: %.3f dB\n', std(nmse_diff(:)));
    fprintf('Max NMSE Improvement: %.3f dB\n', max(nmse_diff(:)));
    fprintf('Min NMSE Improvement: %.3f dB\n', min(nmse_diff(:)));

    % Polar-like colored mesh (θ = azim, r = elev)
    azim_rad = deg2rad(azims);
    [TH, R] = meshgrid(azim_rad, elevs);
    X = R .* cos(TH);
    Y = R .* sin(TH);

    fig = figure('Position',[100 100 700 700]);
    ax = axes(fig); %#ok<LAXES>
    pcolor(ax, X, Y, nmse_diff); shading flat; axis equal tight;
    clim = max(abs(nmse_diff(:))); caxis(ax, [-clim, clim]);
    colorbar(ax); xlabel(ax, 'X'); ylabel(ax, 'Y');
    title(ax, sprintf('\x0394NMSE | %s taper | %s vs %s | %s (%s) | lr=%.1e', ...
        taper, env0_name, env1_name, func2str(loss_fn), loss_scale, lr));

    % Save figure with similar name
    envs_fname = sprintf('%s_vs_%s', strrep(env0_name,' ','_'), strrep(env1_name,' ','_'));
    name = sprintf('MATLAB_patterns_nmse_%s_%s_%s_%s_lr_%.1e.png', taper, envs_fname, func2str(loss_fn), loss_scale, lr);
    exportgraphics(fig, name, 'Resolution', 200);
end

function res = run_optimization(env0_name, env1_name, taper, elev_deg, azim_deg, loss_fn, loss_scale, lr, do_plot)
    if nargin < 9, do_plot = true; end
    fprintf('Running optimization for elev %g°, azim %g°, %s taper\n', elev_deg, azim_deg, taper);

    params = init_params(env0_name, env1_name, taper, elev_deg, azim_deg);
    w = params.w;
    aeps_env1 = params.aeps_env1;
    power_env0 = params.power_env0;

    power_db_opt = optimize(w, aeps_env1, power_env0, loss_fn, loss_scale, lr);

    power_db_env0 = to_db(params.power_env0);
    power_db_env1 = to_db(params.power_env1);

    % Normalize by mean square of target in dB (match Python)
    norm_factor = mean(power_db_env0(:).^2);
    mse_env1_raw = mean((power_db_env1(:) - power_db_env0(:)).^2);
    mse_opt_raw  = mean((power_db_opt(:)  - power_db_env0(:)).^2);

    nmse_env1 = 10*log10(mse_env1_raw / norm_factor);
    nmse_opt  = 10*log10(mse_opt_raw  / norm_factor);

    if do_plot
        fig = figure('Position',[100 100 1200 900]);
        tiledlayout(fig, 3, 1, 'Padding','compact');

        nexttile; plot_power_db(power_db_env0, 'No Env');
        nexttile; plot_power_db(power_db_env1, sprintf('Env 1 | NMSE %.3fdB', nmse_env1));
        nexttile; plot_power_db(power_db_opt,  sprintf('Optimized | NMSE %.3fdB', nmse_opt));

        steer_title = sprintf('Steering Elev %d°, Azim %d°', round(elev_deg), round(azim_deg));
        sgtitle(sprintf('%s taper | %s', taper, steer_title));

        name = sprintf('patterns_%s_elev_%d_azim_%d.png', taper, round(elev_deg), round(azim_deg));
        exportgraphics(fig, name, 'Resolution', 200);
        fprintf('Saved figure to %s\n', name);
    end

    res = struct('power_db_opt', power_db_opt, 'nmse_env1', nmse_env1, 'nmse_opt', nmse_opt);
end

% ========================= Optimization =========================
function power_db_opt = optimize(w_init, aeps, target_power, loss_fn, loss_scale, lr)
    w = w_init; % complex (n_x, n_y)
    nsteps = 100;
    for step = 0:nsteps-1
        [w, loss] = train_step(w, aeps, target_power, lr, loss_fn, loss_scale);
        if mod(step,10) == 0
            fprintf('step %d, loss: %.3f\n', step, real(loss));
        end
    end
    fprintf('Weight norm: %.3f\n', norm(w(:)));

    pattern_opt = synthesize_field(aeps, w);      % (t,p,z)
    power_db_opt = to_db(to_power(pattern_opt));  % (t,p)
end

function [w_next, loss_val] = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)
    % Compute prediction
    field = synthesize_field(aeps, w);   % (t,p,z)
    pred_power = to_power(field);        % (t,p)

    % Potentially compute in dB for the loss
    if strcmpi(loss_scale, 'db')
        pred_eval = to_db(pred_power);
        target_eval = to_db(target_power);
    else
        pred_eval = pred_power;
        target_eval = target_power;
    end

    loss_val = loss_fn(pred_eval, target_eval);

    % Compute gradient wrt w* (Wirtinger) analytically
    % Shapes: aeps: (nx,ny,t,p,z), field: (t,p,z), pred_power: (t,p)
    E = pred_eval - target_eval;   % (t,p)
    N = numel(E);
    if strcmpi(loss_scale, 'db')
        % d/dP [10 log10(P)] = 10 / (ln(10) * P)
        alpha = 10 / log(10);
        dL_dP = (2/N) * (E .* (alpha ./ max(pred_power, realmin)));
    else
        dL_dP = (2/N) * E;
    end

    % Accumulate gradient over t,p,z: G(x,y) = sum_{t,p,z} dL_dP(tp) * s(tpz) * conj(aeps(x,y,tpz))
    G_tpz = dL_dP .* field;             % (t,p,z) via scalar expansion
    % Use implicit expansion to avoid large replications
    % conj(aeps): (nx,ny,t,p,z); reshape(G_tpz) -> (1,1,t,p,z)
    G_full = conj(aeps) .* reshape(G_tpz, [1 1 size(G_tpz,1) size(G_tpz,2) size(G_tpz,3)]);
    G = squeeze(sum(sum(sum(G_full, 5), 4), 3));  % (nx,ny)

    % Gradient step (gradient wrt conjugate)
    w_next = w - lr * G;
end

% ========================= Parameters, Data =========================
function params = init_params(env0_name, env1_name, taper, elev_deg, azim_deg)
    % Root directory is one level up from this file's folder
    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    root_dir = fileparts(this_dir);

    cst_dir = fullfile(root_dir, 'cst');
    aeps_env0 = load_cst_dir(fullfile(cst_dir, env0_name)); % (nx,ny,t,p,z)

    [nx, ny, ~, ~, ~] = size(aeps_env0);
    if strcmpi(taper, 'hamming')
        amplitude = hamming_taper(nx, ny);
    else
        amplitude = uniform_taper(nx, ny);
    end
    amplitude = amplitude ./ sqrt(sum(amplitude(:).^2)); % normalize power to 1

    phase = ideal_steering(nx, ny, 0.5, 0.5, elev_deg, azim_deg);
    w = amplitude .* exp(1j * phase);

    power_env0 = to_power(synthesize_field(aeps_env0, w));

    aeps_env1 = load_cst_dir(fullfile(cst_dir, env1_name));
    power_env1 = to_power(synthesize_field(aeps_env1, w));

    params = struct();
    params.taper = taper;
    params.elev_deg = elev_deg;
    params.azim_deg = azim_deg;
    params.w = w;
    params.aeps_env0 = aeps_env0;
    params.aeps_env1 = aeps_env1;
    params.power_env0 = power_env0;
    params.power_env1 = power_env1;
end

% ========================= Math helpers =========================
function y = to_power(field)
    % Sum over polarization: field (t,p,z) -> power (t,p)
    y = sum(abs(field).^2, 3);
end

function y = to_db(x)
    % Convert to dB with numerical floor
    y = 10 * log10(max(x, realmin));
end

function loss = mse_loss(pred_power, target_power)
    diff = pred_power - target_power;
    loss = mean(diff(:).^2);
end

function idx = argmax_nd(x)
    [~, ind] = max(x(:));
    sz = size(x);
    c = cell(1, numel(sz));
    [c{:}] = ind2sub(sz, ind);
    idx = cellfun(@(v) v, c);
end

% ========================= Plotting =========================
function plot_power_db(power_db, titleStr)
    if nargin < 2, titleStr = ''; end
    % A simple multi-view: 2D and 3D
    t = linspace(0, pi, size(power_db,1));
    p = linspace(0, 2*pi, size(power_db,2));

    % 2D image
    subplot(1,3,1);
    imagesc(rad2deg(t), rad2deg(p), power_db.'); axis xy;
    xlabel('\theta (deg)'); ylabel('\phi (deg)'); title('2D'); colorbar;

    % Sine-space contour
    subplot(1,3,2);
    [TT, PP] = ndgrid(t, p);
    U = sin(TT).*cos(PP); V = sin(TT).*sin(PP);
    contourf(U, V, power_db, 64, 'LineStyle','none'); axis equal tight;
    title('Sine-Space'); colorbar;

    % 3D surface
    subplot(1,3,3);
    R = (power_db - min(power_db(:))) / max(eps, (max(power_db(:)) - min(power_db(:))));
    R = 1 + R; % scale > 1 for nicer viz
    X = R .* sin(TT) .* cos(PP);
    Y = R .* sin(TT) .* sin(PP);
    Z = R .* cos(TT);
    surf(X, Y, Z, power_db, 'EdgeColor','none'); view(30, 20);
    axis vis3d; title('3D'); colorbar;
    sgtitle(titleStr);
end

% ========================= Array synthesis =========================
function field = synthesize_field(aeps, w)
    % aeps: (nx,ny,t,p,z); w: (nx,ny) -> field: (t,p,z)
    % Multiply and sum over x,y
    % Use implicit expansion: (nx,ny,t,p,z) .* (nx,ny)
    S = aeps .* w;                       % (nx,ny,t,p,z)
    S = sum(S, 1); S = sum(S, 2);        % (1,1,t,p,z)
    field = squeeze(S);                  % (t,p,z)
end

% ========================= Data loading (CST) =========================
function aeps = load_cst(cst_dir)
    % Load CST antenna pattern data from a directory
    d = dir(fullfile(cst_dir, '*_RG.txt'));
    if isempty(d)
        error('No CST files matching *_RG.txt found in %s', cst_dir);
    end

    % Extract indices from names like ...[N]_RG.txt and sort
    idx = zeros(numel(d),1);
    for k = 1:numel(d)
        nm = d(k).name;
        % Extract text in brackets [...]
        t = regexp(nm, '\\[(\d+)\\]', 'tokens', 'once');
        if isempty(t)
            error('File %s missing bracketed index', nm);
        end
        idx(k) = str2double(t{1}) - 1; % zero-based like Python
    end
    [~, order] = sort(idx);
    d = d(order);

    % Load each file into (t,p,z)
    fields = cell(numel(d),1);
    for k = 1:numel(d)
        fields{k} = load_cst_file(fullfile(d(k).folder, d(k).name));
    end
    fields = cat(4, fields{:}); % (t,p,z, n_elems)
    % Reshape to (n_elements, t,p,z)
    fields = permute(fields, [4 1 2 3]);

    % Arrange into (4,4,t,p,z) then transpose to (n_x,n_y,t,p,z)
    if size(fields,1) ~= 16
        error('Expected 16 elements; got %d', size(fields,1));
    end
    fields = reshape(fields, [4 4 size(fields,2) size(fields,3) size(fields,4)]); % (4,4,t,p,z)
    aeps = permute(fields, [2 1 3 4 5]); % (n_x, n_y, t, p, z)
end

function field = load_cst_file(cst_path)
    % Load CST antenna pattern data from a file.
    % Columns: elev_deg, azim_deg, abs_grlz, abs_cross, phase_cross_deg,
    %          abs_copol, phase_copol_deg, ax_ratio
    opts = detectImportOptions(cst_path, 'NumHeaderLines', 2);
    tbl = readmatrix(cst_path, opts); % numeric matrix
    if size(tbl,2) < 8
        tbl = readmatrix(cst_path, 'NumHeaderLines', 2);
    end
    data = tbl(:, 1:8);

    % Sort by elev_deg then azim_deg (as in Python)
    data = sortrows(data, [1 2]);

    phase_cross = deg2rad(data(:,5));
    phase_copol = deg2rad(data(:,7));
    E_cross = sqrt(data(:,4)) .* exp(1j * phase_cross);
    E_copol = sqrt(data(:,6)) .* exp(1j * phase_copol);
    field = cat(3, E_cross, E_copol);  % (N, 2)

    % Reshape to (elev, azim, n_pol) => (181, 360, 2), then drop last elev
    try
        field = reshape(field, [181, 360, 2]);
    catch
        error('Unexpected CST file length in %s; cannot reshape to 181x360x2', cst_path);
    end
    field = field(1:end-1, :, :); % remove last elev value => (180,360,2)
end

function aeps = load_cst_dir(cst_dir)
    % Robust CST loader that handles names like 'farfield (f=...) [10]_RG.txt'
    d = dir(fullfile(cst_dir, '*_RG*.txt'));
    if isempty(d)
        error('No CST files matching *_RG*.txt found in %s', cst_dir);
    end

    % Extract indices from bracket [N] or before _RG
    idx = zeros(numel(d),1);
    for k = 1:numel(d)
        nm = d(k).name;
        t = regexp(nm, '\[(\d+)\]', 'tokens', 'once');
        if isempty(t)
            t = regexp(nm, '(\d+)\s*_RG', 'tokens', 'once');
        end
        if isempty(t)
            error('File %s missing bracketed index', nm);
        end
        if iscell(t), token = t{1}; else, token = t; end
        idx(k) = str2double(token) - 1;
    end
    [~, order] = sort(idx);
    d = d(order);

    fields = cell(numel(d),1);
    for k = 1:numel(d)
        fields{k} = load_cst_file(fullfile(d(k).folder, d(k).name));
    end
    fields = cat(4, fields{:});           % (t,p,z, n_elems)
    fields = permute(fields, [4 1 2 3]);  % (n_elems, t,p,z)

    if size(fields,1) ~= 16
        error('Expected 16 elements; got %d', size(fields,1));
    end
    fields = reshape(fields, [4 4 size(fields,2) size(fields,3) size(fields,4)]); % (4,4,t,p,z)
    aeps = permute(fields, [2 1 3 4 5]); % (n_x, n_y, t, p, z)
end

% ========================= Tapering and steering =========================
function amplitude = uniform_taper(nx, ny)
    amplitude = ones(nx, ny);
end

function amplitude = hamming_taper(nx, ny)
    amplitude = hamming(nx) * hamming(ny).';
end

function phase = ideal_steering(nx, ny, dx, dy, elev_deg, azim_deg)
    % Spacing in wavelengths (dx, dy)
    elev_rad = deg2rad(elev_deg);
    azim_rad = deg2rad(azim_deg);
    sin_elev = sin(elev_rad);
    sin_azim = sin(azim_rad); cos_azim = cos(azim_rad);

    i = (0:nx-1) - (nx-1)/2;
    j = (0:ny-1) - (ny-1)/2;
    [ii, jj] = ndgrid(i, j);

    kd_x = 2*pi*dx*ii;
    kd_y = 2*pi*dy*jj;
    phase = kd_x .* sin_elev .* cos_azim + kd_y .* sin_elev .* sin_azim;
end
