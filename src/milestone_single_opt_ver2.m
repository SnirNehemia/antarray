function milestone_single_opt_ver2()
% Single-optimization MATLAB port matching src/milestone.py behavior.
% - Loads CST element patterns for two environments
% - Builds tapered/steered initial weights
% - Optimizes against target power pattern (env0) using MSE
% - Plots 3x3 layout: No Env, Env1, Optimized (2D, sine-space, 3D)
% - Saves figure as patterns_<taper>_elev_<E>_azim_<A>.png

    % Configuration (match your Python runs)
    env0_name = 'no_env_rotated';
    env1_name = 'Env1_rotated';
    taper = 'hamming';              % 'hamming' or 'uniform'
    elev_deg = 10;                  % pointing elevation (deg)
    azim_deg = 10;                  % pointing azimuth (deg)
    loss_scale = 'linear';          % 'linear' or 'db'
    lr = 1e-6;                      % learning rate

    loss_fn = @mse_loss;
    tic()
    % Run single optimization and plot
    run_optimization(env0_name, env1_name, taper, elev_deg, azim_deg, loss_fn, loss_scale, lr, true);
    fprintf('total runtime is: %.0f sec = %.1f min\n', toc(), toc()/60);
    fprintf('done!\n')
end

function [res, w] = run_optimization(env0_name, env1_name, taper, elev_deg, azim_deg, loss_fn, loss_scale, lr, do_plot)
    if nargin < 10, do_plot = true; end
    fprintf('Running optimization for elev %g°, azim %g°, %s taper\n', elev_deg, azim_deg, taper);

    params = init_params(env0_name, env1_name, taper, elev_deg, azim_deg);
    w = params.w;

    [power_db_opt, w] = optimize(w, params.aeps_env1, params.power_env0, loss_fn, loss_scale, lr);

    power_db_env0 = to_db(params.power_env0);
    power_db_env1 = to_db(params.power_env1);

    % NMSE in dB with target in dB, normalized by mean square of target (dB)
    norm_factor = mean(power_db_env0(:).^2);
    mse_env1_raw = mean((power_db_env1(:) - power_db_env0(:)).^2);
    mse_opt_raw  = mean((power_db_opt(:)  - power_db_env0(:)).^2);
    nmse_env1 = 10*log10(mse_env1_raw / norm_factor);
    nmse_opt  = 10*log10(mse_opt_raw  / norm_factor);

    if do_plot
        fig = figure('Position',[100 100 1500 1000]);
        tl = tiledlayout(fig, 3, 3, 'Padding','compact', 'TileSpacing','compact');

        draw_pattern_row(tl, 1, power_db_env0, 'No Env');
        draw_pattern_row(tl, 2, power_db_env1, sprintf('Env 1 | NMSE %.3fdB', nmse_env1));
        draw_pattern_row(tl, 3, power_db_opt,  sprintf('Optimized | NMSE %.3fdB', nmse_opt));

        steer_title = sprintf('Steering Elev %d°, Azim %d°', round(elev_deg), round(azim_deg));
        title(tl, sprintf('%s taper | %s', taper, steer_title));

        name = sprintf('MATLAB_patterns_%s_elev_%d_azim_%d.png', taper, round(elev_deg), round(azim_deg));
        exportgraphics(fig, name, 'Resolution', 200);
        fprintf('Saved figure to %s\n', name);
    end

    res = struct('power_db_opt', power_db_opt, 'nmse_env1', nmse_env1, 'nmse_opt', nmse_opt);
end

% ========================= Optimization =========================
function [power_db_opt, w] = optimize(w_init, aeps, target_power, loss_fn, loss_scale, lr)
    w = w_init; % complex (n_x, n_y)
    nsteps = 100;
    for step = 0:nsteps-1
        [w, loss] = train_step(w, aeps, target_power, lr, loss_fn, loss_scale);
        if mod(step,10) == 0
            fprintf('step %d, loss: %.3f\n', step, real(loss));
        end
    end
    fprintf('Weight norm: %.3f\n', norm(w(:)));

    pattern_opt = synthesize_field(aeps, w);      % (theta, phi, pol)
    power_db_opt = to_db(to_power(pattern_opt));  % (theta, phi)
end

function [w_next, loss_val] = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)
    % Forward
    field = synthesize_field(aeps, w);   % (t,p,z)
    pred_power = to_power(field);        % (t,p)

    if strcmpi(loss_scale, 'db')
        pred_eval = to_db(pred_power);
        target_eval = to_db(target_power);
    else
        pred_eval = pred_power;
        target_eval = target_power;
    end

    loss_val = loss_fn(pred_eval, target_eval);

    % Wirtinger gradient wrt conj(w)
    E = pred_eval - target_eval;   % (t,p)
    N = numel(E);
    if strcmpi(loss_scale, 'db')
        alpha = 10 / log(10);
        dL_dP = (2/N) * (E .* (alpha ./ max(pred_power, realmin)));
    else
        dL_dP = (2/N) * E;
    end

    G_tpz = dL_dP .* field; % (t,p,z)
    G_full = conj(aeps) .* reshape(G_tpz, [1 1 size(G_tpz,1) size(G_tpz,2) size(G_tpz,3)]);
    G = squeeze(sum(sum(sum(G_full, 5), 4), 3));  % (nx,ny)

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
    y = sum(abs(field).^2, 3); % (t,p)
end

function y = to_db(x)
    y = 10 * log10(max(x, realmin));
end

function loss = mse_loss(pred_power, target_power)
    diff = pred_power - target_power;
    loss = mean(diff(:).^2);
end

% ========================= Plotting helpers =========================
function draw_pattern_row(tl, row_idx, pattern_db, label)
    % Normalize orientation to (theta, phi)
    [Ztp, theta_deg, phi_deg] = ensure_theta_phi(pattern_db);

    % 2D
    axA = nexttile(tl, (row_idx-1)*3 + 1);
    imagesc(axA, phi_deg, theta_deg, Ztp); set(axA,'YDir','normal');
    xlabel(axA,'\phi (deg)'); ylabel(axA,'\theta (deg)'); title(axA, [label ' | 2D']); colorbar(axA);

    % Sine-space
    axB = nexttile(tl, (row_idx-1)*3 + 2);
    [TT, PP] = ndgrid(deg2rad(theta_deg), deg2rad(phi_deg));
    U = sin(TT).*cos(PP); V = sin(TT).*sin(PP);
    contourf(axB, U, V, Ztp, 128, 'LineStyle','none'); axis(axB,'equal'); axis(axB,'tight');
    title(axB, [label ' | Sine-Space']); colorbar(axB);

    % 3D
    axC = nexttile(tl, (row_idx-1)*3 + 3);
    patt = Ztp; patt = (patt - min(patt(:))) / max(eps, (max(patt(:)) - min(patt(:)))); patt = 2*patt;
    X = patt .* sin(TT) .* cos(PP);
    Y = patt .* sin(TT) .* sin(PP);
    Z = patt .* cos(TT);
    surf(axC, X, Y, Z, Ztp, 'EdgeColor','none'); view(axC, 30, 20); axis(axC,'vis3d');
    title(axC, [label ' | 3D']); colorbar(axC);
end

% Backwards-compat plotting function (unused now)
function plot_ff_2d(ax, pattern_db, label)
    [Ztp, theta_deg, phi_deg] = ensure_theta_phi(pattern_db);
    imagesc(ax, phi_deg, theta_deg, Ztp); set(ax,'YDir','normal');
    xlabel(ax,'\phi (deg)'); ylabel(ax,'\theta (deg)'); title(ax, sprintf('%s | 2D', label)); colorbar(ax);
end

function plot_sine_space(ax, pattern_db, label)
    [Ztp, theta_deg, phi_deg] = ensure_theta_phi(pattern_db);
    [TT, PP] = ndgrid(deg2rad(theta_deg), deg2rad(phi_deg));
    U = sin(TT).*cos(PP); V = sin(TT).*sin(PP);
    contourf(ax, U, V, Ztp, 128, 'LineStyle','none');
    axis(ax,'equal'); axis(ax,'tight');
    title(ax, sprintf('%s | Sine-Space', label)); colorbar(ax);
end

function plot_ff_3d(ax, pattern_db, label)
    [Ztp, theta_deg, phi_deg] = ensure_theta_phi(pattern_db);
    [TT, PP] = ndgrid(deg2rad(theta_deg), deg2rad(phi_deg));
    patt = Ztp; patt = (patt - min(patt(:))) / max(eps, (max(patt(:)) - min(patt(:)))); patt = 2*patt;
    X = patt .* sin(TT) .* cos(PP);
    Y = patt .* sin(TT) .* sin(PP);
    Z = patt .* cos(TT);
    surf(ax, X, Y, Z, Ztp, 'EdgeColor','none'); view(ax, 30, 20); axis(ax,'vis3d');
    title(ax, sprintf('%s | 3D', label)); colorbar(ax);
end

% ========================= Orientation helper =========================
function [Ztp, theta_deg, phi_deg] = ensure_theta_phi(Z)
    % Ensure pattern is (theta, phi). If it's (phi, theta), transpose.
    sz = size(Z);
    if numel(sz) ~= 2
        error('Pattern must be 2D, got size %s', mat2str(sz));
    end
    if isequal(sz, [180 360])
        Ztp = Z; theta_deg = 0:179; phi_deg = 0:359; return; %#ok<RETURN>
    elseif isequal(sz, [360 180])
        Ztp = Z.'; theta_deg = 0:179; phi_deg = 0:359; return; %#ok<RETURN>
    end

    % Fallback heuristic: choose orientation that makes theta=0 nearly constant across phi
    if sz(1) < sz(2)
        theta_deg = 0:(sz(1)-1); phi_deg = 0:(sz(2)-1);
        row_std = std(Z(1,:)); col_std = std(Z(:,1));
        if row_std < col_std
            Ztp = Z; return; %#ok<RETURN>
        else
            Ztp = Z.'; theta_deg = 0:(sz(2)-1); phi_deg = 0:(sz(1)-1); return; %#ok<RETURN>
        end
    else
        theta_deg = 0:(sz(2)-1); phi_deg = 0:(sz(1)-1);
        row_std = std(Z(:,1)); col_std = std(Z(1,:));
        if row_std < col_std
            Ztp = Z.'; return; %#ok<RETURN>
        else
            Ztp = Z; theta_deg = 0:(sz(1)-1); phi_deg = 0:(sz(2)-1); return; %#ok<RETURN>
        end
    end
end

% ========================= Array synthesis =========================
function field = synthesize_field(aeps, w)
    % aeps: (nx,ny,theta,phi,pol); w: (nx,ny) -> field: (theta,phi,pol)
    S = aeps .* w;                       % (nx,ny,t,p,z)
    S = sum(S, 1); S = sum(S, 2);        % (1,1,t,p,z)
    field = squeeze(S);                  % (t,p,z)
end

% ========================= CST loading =========================
function aeps = load_cst_dir(cst_dir)
    % Robust CST loader: matches '*_RG*.txt' and sorts by [N] or digits before _RG
    d = dir(fullfile(cst_dir, '*_RG*.txt'));
    if isempty(d)
        error('No CST files matching *_RG*.txt found in %s', cst_dir);
    end

    idx = zeros(numel(d),1);
    for k = 1:numel(d)
        nm = d(k).name;
        t = regexp(nm, '\[(\d+)\]', 'tokens', 'once');
        if isempty(t)
            t = regexp(nm, '(\d+)\s*_RG', 'tokens', 'once');
        end
        if isempty(t)
            error('File %s missing element index', nm);
        end
        if iscell(t), token = t{1}; else, token = t; end
        idx(k) = str2double(token) - 1; % zero-based for sorting
    end
    [~, order] = sort(idx);
    d = d(order);

    fields = cell(numel(d),1);
    for k = 1:numel(d)
        fields{k} = load_cst_file(fullfile(d(k).folder, d(k).name)); % (t,p,z)
    end
    fields = cat(4, fields{:});           % (t,p,z, n_elems)
    fields = permute(fields, [4 1 2 3]);  % (n_elems, t,p,z)

    if size(fields,1) ~= 16
        error('Expected 16 elements; got %d', size(fields,1));
    end
    fields = reshape(fields, [4 4 size(fields,2) size(fields,3) size(fields,4)]); % (4,4,t,p,z)
    aeps = permute(fields, [2 1 3 4 5]); % (n_x, n_y, t, p, z)
end

function field = load_cst_file(cst_path)
    % Load CST element pattern from a single file.
    % Columns: elev_deg, azim_deg, abs_grlz, abs_cross, phase_cross_deg,
    %          abs_copol, phase_copol_deg, ax_ratio
    opts = detectImportOptions(cst_path, 'NumHeaderLines', 2);
    tbl = readmatrix(cst_path, opts);
    if size(tbl,2) < 8
        tbl = readmatrix(cst_path, 'NumHeaderLines', 2);
    end
    data = tbl(:, 1:8);

    % Sort by elev then azim as in Python
    data = sortrows(data, [1 2]);

    phase_cross = deg2rad(data(:,5));
    phase_copol = deg2rad(data(:,7));
    E_cross = sqrt(data(:,4)) .* exp(1j * phase_cross);
    E_copol = sqrt(data(:,6)) .* exp(1j * phase_copol);
    field = cat(3, E_cross, E_copol);  % (N, 2)

    % Reshape to (elev, azim, pol) = (181,360,2), drop last elev to 180
    try
        field = reshape(field, [181, 360, 2]);
    catch
        error('Unexpected CST length in %s; cannot reshape to 181x360x2', cst_path);
    end
    field = field(1:end-1, :, :); % (180,360,2)
end

% ========================= Tapering / Steering =========================
function amplitude = uniform_taper(nx, ny)
    amplitude = ones(nx, ny);
end

function amplitude = hamming_taper(nx, ny)
    amplitude = hamming(nx) * hamming(ny).';
end

function phase = ideal_steering(nx, ny, dx, dy, elev_deg, azim_deg)
    % Spacing (dx,dy) in wavelengths
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
