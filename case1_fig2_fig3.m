% This script implements a unified neuromorphic controller for a car-pendulum system.
% It utilizes Dual Half-Center Oscillators (HCOs) to achieve both swing-up (via energy pumping)
% and stabilization (via error-based modulation) using a single control framework.
%
% Author: Xinxin Zhang
% Date: 2025-12-01

%% ================================================================================
%% UNIFIED NEUROMORPHIC CAR-PENDULUM CONTROLLER (POSITION-CONDITIONED STABILIZATION)
%% ================================================================================
% This script implements a fully UNIFIED Neuromorphic Controller based on 
% Dual Half-Center Oscillators (HCOs) for both:
% 1. Swing-up (Energy Pumping)
% 2. Stabilization (Error-based input modulation)
%
% The control output is a HORIZONTAL FORCE (F) applied to the CART.
 
% Date: 2025-11-23 (Car-Pendulum Conversion)
%% ================================================================================
clear; clc; close all;
%% SECTION 1: REFERENCE SIGNAL GENERATION
%% ================================================================================
Tf = 30; % Total simulation time (s)
f_theta = 10;
dt = 1/f_theta/1e2;
t = 0:dt:Tf;
theta_ref = pi * ones(size(t));
% Assuming cart position reference is 0 for stabilization
x_ref = zeros(size(t)); 
theta_interp = @(tq) interp1(t, theta_ref, tq, 'nearest', 'extrap');
dot_theta_ref = @(tq) 0; % Assuming zero angular velocity reference
%% ================================================================================
%% SECTION 2: CAR-PENDULUM PHYSICAL PARAMETERS
%% ================================================================================
% --- Cart Parameters ---
Mc = 1.0;           % Mass of the Cart (kg)
bc = 0.5;           % Viscous Damping on Cart (N-s/m)
% --- Pendulum Link Parameters ---
m = 0.024;          % Mass of Pendulum Link (kg) - 'm'
L = 0.129;          % Total Length of Pendulum Link (m) - 'L'
b = 0.00005;        % Viscous Damping of Pendulum Link - 'Damp' (was 'b')
g = 9.81;           % Gravitational acceleration (m/s²)
L_half = L/2;       % Distance to Center of Mass
Jperp = (1/3) * m * L^2; % Moment of inertia about the pivot (J)
% Target energy for the upright position (E = m * g * L)
E_target = m * g * L; 
fprintf('Using Car-Pendulum Dynamics:\n');
fprintf('  Cart Mass (Mc): %.3f kg, Pendulum Mass (m): %.3f kg\n', Mc, m);
fprintf('Target mechanical energy for upright position (theta=pi): %.3f J\n', E_target);
%% ================================================================================
%% SECTION 3: NEUROMORPHIC CONTROLLER PARAMETER TUNING
%% ================================================================================
% --- Neuromorphic Parameters ---
f_base = 1;
tau_f_base = 0.1;
tau_s_base = 20*tau_f_base;
tau_us_base = 100*tau_f_base;
tau_f = tau_f_base * f_base / f_theta;
tau_s = tau_s_base * f_base / f_theta;
tau_us = tau_us_base * f_base / f_theta;
force_gain = 5; % Initial Force gain (Replaced torque_gain)
I_ext_base = -1.0; % Baseline drive for Swing-up (Oscillation)
threshold = 0;
% HCO Parameters (Unchanged)
g_f = 2; 
g_splus = 1.5; 
g_sminus = 1; 
g_us = 1.5; 
a_cross = 1.5;
synapse = @(vs,gain) gain./(1 + exp(-2*(vs+1)));
% --- Adaptation & Feedback Parameters ---
learning_rate_energy = 1;
Ki_gain = 0.01;
% Gains for stabilization feedback (applied as FORCE on the cart)
K_error_feedback = 100; 
K_velocity_feedback = 20;
% Maximum FORCE (Replaced max_abs_torque)
max_abs_force = 5; % N (Adjusted for cart-pole force input) 
% --- Position-based Stabilization Parameters ---
theta_tolerance = deg2rad(5); % +/- 5 degrees around pi to trigger stabilization (Increased for Cart-Pole)
I_ext_stabilization = 2;     % Positive I_ext value for "quiet" stabilization
fprintf('Maximum controller force applied: %.2f N\n', max_abs_force);
fprintf('Stabilization mode activates when |theta - pi| < %.2f rad (%.1f deg)\n', theta_tolerance, rad2deg(theta_tolerance));
% --- MODIFIED: Sinusoidal Disturbance Parameters (Force on Cart) ---
A_disturb = 1.0;       % Amplitude of the sine wave (N)
f_disturb = 2.0;       % Frequency of the sine wave (Hz)
t_disturb_start = Tf/2; % Time (s) to start disturbance
t_disturb_end = Tf;   % Time (s) to end disturbance
fprintf('Sinusoidal disturbance (A=%.1f N, f=%.1f Hz) applied between t=%.1f s and t=%.1f s.\n', A_disturb, f_disturb, t_disturb_start, t_disturb_end);
%% ================================================================================
%% SECTION 4: SIMULATION: UNIFIED NEUROMORPHIC CONTROL (POSITION-CONDITIONED)
%% ================================================================================
fprintf('\nStarting Simulation: Car-Pendulum with Neuromorphic Controller...\n');
% State vector: [v1...v12 (NM: 12 states); x (13); theta (14); dx (15); dtheta (16)] = 16 states
x_unified = zeros(16, length(t));
% Initial conditions: NM states; x=0; theta=0.1; dx=0; dtheta=0
x_unified(:,1) = [zeros(1, 12), 0, 0.1, 0, 0]; 
x_unified(3,1) = -1; % Initial condition for v_s1 to break symmetry
x_unified(9,1) = -1; % Initial condition for v_s2 to break symmetry
% Initialize adaptive gains and integral term
force_pos_gain = force_gain; % Used to be torque_pos_gain
force_neg_gain = force_gain; % Used to be torque_neg_gain
energy_error_integral = 0;
current_I_ext = I_ext_base;
% Logging variables
x_hist = zeros(size(t));
dx_hist = zeros(size(t));
theta_hist = zeros(size(t));
omega_hist = zeros(size(t));
force_hist = zeros(size(t)); % Logs the final applied Force F (clamped + disturbance)
raw_nm_force_hist = zeros(size(t)); % NEW: Logs raw NM force before clamping/disturbance
E_mechanical_hist = zeros(size(t));
control_mode_hist = zeros(size(t)); 
disturb_hist = zeros(size(t)); 
% Log the controller force BEFORE disturbance is added (used for effort analysis)
F_controller_history = zeros(size(t)); 
% --- NEW: Acceleration History Initialization ---
ddot_x_hist = zeros(size(t)); 
% ------------------------------------------------
% Log initial state
x_hist(1) = x_unified(13,1);
theta_hist(1) = x_unified(14,1);
dx_hist(1) = x_unified(15,1);
omega_hist(1) = x_unified(16,1);
% Energy calculation for logging
KE_p = 0.5 * Jperp * x_unified(16,1)^2;
PE_p = m * g * L_half * (1 - cos(x_unified(14,1)));
KE_c = 0.5 * Mc * x_unified(15,1)^2;
E_mechanical_hist(1) = KE_p + PE_p + KE_c;
control_mode_hist(1) = 1; % Start in Swing-up
for k = 2:length(t)
    % Extract physical states from the unified vector
    x = x_unified(13,k-1);
    theta = x_unified(14,k-1);
    dx = x_unified(15,k-1);
    omega_p = x_unified(16,k-1); % dtheta
    
    % Angular error relative to the target inverted position (pi)
    theta_error = wrapToPi(theta_ref(k-1) - theta);
    dot_theta_error = dot_theta_ref(t(k-1)) - omega_p;
    
    % --- Energy Calculation (Pendulum-Only Mechanical Energy) ---
    KE_p = 0.5 * Jperp * omega_p^2; 
    PE_p = m * g * L_half * (1 - cos(theta)); 
    E_current = KE_p + PE_p;
    energy_error = E_target - E_current;
    
    % --- Position-Based Switching Condition ---
    is_stabilizing = abs(theta_error) < theta_tolerance;
    
    % --- Adaptive Gain & I_ext Logic ---
    if is_stabilizing 
        % Stabilization Phase (Quiet control by setting I_ext positive)
        current_I_ext = I_ext_stabilization; 
        
        % Disable adaptive energy gain, fix the force gains to a low value
        force_pos_gain =1; % Fixed base gain for stabilization
        force_neg_gain = 1; % Fixed base gain for stabilization
        gain_update = 0; 
        
        control_mode_hist(k) = 2; % Log Stabilization
    else
        % Swing-up Phase (Energy Pumping)
        current_I_ext = I_ext_base;
        
        % Calculate adaptive gain (Energy Pumping)
        energy_error_integral = energy_error_integral + energy_error * dt;
        gain_update = learning_rate_energy * energy_error + Ki_gain * energy_error_integral;
        
        force_pos_gain = force_pos_gain + gain_update * dt;
        force_neg_gain = force_neg_gain - gain_update * dt;
        
        % Ensure gains are positive and reasonable
        max_adaptive_gain =5; % Allow higher gain for swing-up force
        force_pos_gain = max(min(force_pos_gain, max_adaptive_gain), 1e-4);
        force_neg_gain = max(min(force_neg_gain, max_adaptive_gain), 1e-4);
        
        control_mode_hist(k) = 1; % Log Swing-up
    end
    
    % --- 1. Neuromorphic Force Calculation (Always Active) ---
    hco1_output = (1./(1 + exp(-2*(x_unified(1, k-1) - threshold))));
    hco2_output = (1./(1 + exp(-2*(x_unified(7, k-1) - threshold))));
    
    % The raw NM output (Swing-up / Stabilization action)
    force_nm = force_pos_gain * hco1_output - force_neg_gain * hco2_output;
    raw_nm_force_hist(k) = force_nm; % LOG RAW FORCE
    
    F_controller = force_nm; 
    
    % --- 2. Unified External Input Modulation (The Stabilization) ---
    % Combined angular and velocity error feedback (Proportional-Derivative-like)
    error_feedback = K_error_feedback * theta_error + K_velocity_feedback * dot_theta_error;
    
    % External inputs to the HCOs:
    input_hco1 = current_I_ext + max(error_feedback, 0); 
    input_hco2 = current_I_ext + max(-error_feedback, 0); 
    
    % Clamp controller Force for safety
    F_controller = max(min(F_controller, max_abs_force), -max_abs_force);
    
    % --- LOG F_controller (before disturbance) ---
    F_controller_history(k) = F_controller;
    
    % --- MODIFIED: External Sinusoidal Disturbance (Force on Cart) ---
    current_disturb = 0;
    if t(k-1) >= t_disturb_start && t(k-1) < t_disturb_end
        current_disturb = A_disturb * sin(2 * pi * f_disturb * (t(k-1) - t_disturb_start));
    end
    
    % Final Force applied to the system (Controller Output + Disturbance)
    applied_force = F_controller + current_disturb;
    
    % --- DUAL HCO NEUROMORPHIC DYNAMICS (Euler Integration) ---
    dxdt = zeros(16,1);
    
    % HCO Group 1 (Neurons 1-6) - Positive Force Drive
    dxdt(1) = (-x_unified(1,k-1) + g_f*tanh(x_unified(1,k-1)) - g_splus*tanh(x_unified(2,k-1)) + g_sminus*tanh(x_unified(2,k-1)+0.9) - g_us*tanh(x_unified(3,k-1)+0.9) + synapse(x_unified(5,k-1), -0.2) - a_cross*tanh(x_unified(7,k-1)) + input_hco1) / tau_f;
    dxdt(2) = (x_unified(1,k-1) - x_unified(2,k-1)) / tau_s;
    dxdt(3) = (x_unified(1,k-1) - x_unified(3,k-1)) / tau_us;
    dxdt(4) = (-x_unified(4,k-1) + g_f*tanh(x_unified(4,k-1)) - g_splus*tanh(x_unified(5,k-1)) + g_sminus*tanh(x_unified(5,k-1)+0.9) - g_us*tanh(x_unified(6,k-1)+0.9) + synapse(x_unified(2,k-1), -0.2) + a_cross*tanh(x_unified(7,k-1)) + input_hco1) / tau_f;
    dxdt(5) = (x_unified(4,k-1) - x_unified(5,k-1)) / tau_s;
    dxdt(6) = (x_unified(4,k-1) - x_unified(6,k-1)) / tau_us;
    
    % HCO Group 2 (Neurons 7-12) - Negative Force Drive
    dxdt(7) = (-x_unified(7,k-1) + g_f*tanh(x_unified(7,k-1)) - g_splus*tanh(x_unified(8,k-1)) + g_sminus*tanh(x_unified(8,k-1)+0.9) - g_us*tanh(x_unified(9,k-1)+0.9) + synapse(x_unified(11,k-1), -0.2) - a_cross*tanh(x_unified(1,k-1)) + input_hco2) / tau_f;
    dxdt(8) = (x_unified(7,k-1) - x_unified(8,k-1)) / tau_s;
    dxdt(9) = (x_unified(7,k-1) - x_unified(9,k-1)) / tau_us;
    dxdt(10) = (-x_unified(10,k-1) + g_f*tanh(x_unified(10,k-1)) - g_splus*tanh(x_unified(11,k-1)) + g_sminus*tanh(x_unified(11,k-1)+0.9) - g_us*tanh(x_unified(12,k-1)+0.9) + synapse(x_unified(8,k-1), -0.2) + a_cross*tanh(x_unified(1,k-1)) + input_hco2) / tau_f;
    dxdt(11) = (x_unified(10,k-1) - x_unified(11,k-1)) / tau_s;
    dxdt(12) = (x_unified(10,k-1) - x_unified(12,k-1)) / tau_us;
    
    % --- CAR-PENDULUM COUPLED DYNAMICS ---
    % State: [x (13); theta (14); dx (15); dtheta (16)]
    
    % Terms for the coupled equations (from Euler-Lagrange)
    I_p = Jperp;
    
    % A * ddot(X) = B
    % $\begin{pmatrix} M_c + m & m L_{half} \cos\theta \\ m L_{half} \cos\theta & I_p + m L_{half}^2 \end{pmatrix} \begin{pmatrix} \ddot{x} \\ \ddot{\theta} \end{pmatrix} = \begin{pmatrix} F - b_{cart} \dot{x} + m L_{half} \dot{\theta}^2 \sin\theta \\ m g L_{half} \sin\theta - b \dot{\theta} \end{pmatrix}$
    
    A11 = Mc + m;
    A12 = m * L_half * cos(theta);
    B1  = applied_force - bc * dx + m * L_half * omega_p^2 * sin(theta);
    
    A21 = m * L_half * cos(theta);
    A22 = I_p + m * L_half^2;
    B2  = m * g * L_half * sin(theta) - b * omega_p; 
    
    Delta = A11 * A22 - A12 * A21;
    
    ddot_x = (B1 * A22 - B2 * A12) / Delta;
    ddot_theta = (A11 * B2 - A21 * B1) / Delta;
    
    % --- NEW: Store Acceleration for Printing ---
    ddot_x_hist(k) = ddot_x;
    % --------------------------------------------
    % Update physical states
    dxdt(13) = dx;              % x_dot
    dxdt(14) = omega_p;         % theta_dot
    dxdt(15) = ddot_x;          % x_ddot
    dxdt(16) = ddot_theta;      % theta_ddot
    
    % State Update
    x_unified(:,k) = x_unified(:,k-1) + dt * dxdt;
    
    % --- CONSTRAINT: Angle Wrapping ---
    % x_unified(14,k) = wrapToPi(x_unified(14,k)); 
    x_unified(14,k) = (x_unified(14,k)); 
    
    % Data Logging
    x_hist(k) = x_unified(13,k);
    theta_hist(k) = x_unified(14,k);
    dx_hist(k) = x_unified(15,k);
    omega_hist(k) = x_unified(16,k);
    force_hist(k) = applied_force; 
    disturb_hist(k) = current_disturb; 
    
    % Log total mechanical energy (Cart + Pendulum)
    KE_p_log = 0.5 * Jperp * x_unified(16,k)^2;
    PE_p_log = m * g * L_half * (1 - cos(x_unified(14,k)));
    KE_c_log = 0.5 * Mc * x_unified(15,k)^2;
    E_mechanical_hist(k) = KE_p_log + PE_p_log + KE_c_log;
end
% Helper function to wrap angle to [-pi, pi]
function angle_wrapped = wrapToPi(angle)
    angle_wrapped = angle - 2*pi*floor((angle + pi)/(2*pi));
end
%% ================================================================================
%% SECTION 5: STATISTICAL ANALYSIS AND PRINTING
%% ================================================================================
% --- 5.1 Pre-Calculation of Errors and Indices ---
% Errors are calculated against the reference signals (theta_ref = pi, x_ref = 0)
theta_error_full = wrapToPi(theta_ref - theta_hist);
x_error_full = x_ref - x_hist; 
% Find indices for time intervals
t_13s_idx = find(t >= Tf/2-2, 1, 'first');
t_15s_idx = find(t <= Tf/2, 1, 'last');
t_28s_idx = find(t >= Tf-2, 1, 'first');
t_30s_idx = find(t <= Tf, 1, 'last');
% Find the index where the controller first switches to stabilization mode (mode 2)
t_switch_index = find(control_mode_hist == 2, 1, 'first');
% --- 5.2 Overall Performance Metrics (First 15s) ---
max_cart_p = max( x_hist);
max_force = max(abs(force_hist));
% Total Work Done on Cart: integral(|F| * |dx|) dt
force_work_integral = sum(abs(force_hist) .* abs(dx_hist)) * dt; 
fprintf('\n--- Performance Statistics (Neuromorphic Car-Pendulum Control) ---\n');
fprintf('A. Overall Metrics (Tf = %.1f s):\n', Tf);
fprintf('   1. Maximum Applied Force: %.3f m', max_cart_p);
fprintf('   1. Maximum Cart Position: %.3f N\n', max_force);
fprintf('   2. Total Work Done on Cart: %.2f J\n', force_work_integral);
fprintf('   3. Final Cart Position: %.3f m\n', x_hist(end));
fprintf('-------------------------------------------------------------\n');
% --- 5.3 Steady-State Performance (13s - 15s: Pre-Disturbance SS) ---
idx_ss1 = t_13s_idx:t_15s_idx;
theta_err_ss1 = theta_error_full(idx_ss1);
x_err_ss1 = x_error_full(idx_ss1);
theta_max_err_ss1 = max(abs(theta_err_ss1))/pi*100;
theta_rms_err_ss1 = rms(theta_err_ss1)/pi*100;
x_max_err_ss1 = max(abs(x_err_ss1));
x_rms_err_ss1 = rms(x_err_ss1);
fprintf('\nB. Steady-State Performance (Tf/2-2 s - Tf/2 s, Pre-Disturbance):\n');
fprintf('   | State | Max Error | RMS Error |\n');
fprintf('   |-------|-----------|-----------|\n');
fprintf('   | Angle | %9.5f na | %9.5f na |\n', theta_max_err_ss1, theta_rms_err_ss1);
fprintf('   | Cart X| %9.5f m   | %9.5f m   |\n', x_max_err_ss1, x_rms_err_ss1);
% --- 5.4 Disturbance Rejection Performance (28s - 30s: Post-Disturbance SS) ---
idx_ss2 = t_28s_idx:t_30s_idx;
theta_err_ss2 = theta_error_full(idx_ss2);
x_err_ss2 = x_error_full(idx_ss2);
theta_max_err_ss2 = max(abs(theta_err_ss2))/pi*100;
theta_rms_err_ss2 = rms(theta_err_ss2)/pi*100;
x_max_err_ss2 = max(abs(x_err_ss2));
x_rms_err_ss2 = rms(x_err_ss2);
fprintf('\nC. Disturbance Rejection Performance (Tf-2 s - Tf s, Post-Disturbance):\n');
fprintf('   | State | Max Error | RMS Error |\n');
fprintf('   |-------|-----------|-----------|\n');
fprintf('   | Angle | %9.5f na | %9.5f na |\n', theta_max_err_ss2, theta_rms_err_ss2);
fprintf('   | Cart X| %9.5f m   | %9.5f m   |\n', x_max_err_ss2, x_rms_err_ss2);
% --- 5.5 Control Effort and Peak Values ---
% Control Efficiency: Integral of squared control force
% This is a standard metric for control effort (Energy Cost)
J_control_effort = sum(F_controller_history.^2) * dt; 
max_cart_velocity = max(abs(x_unified(15, :))); % Index 15 is dx
max_angular_velocity = max(abs(x_unified(16, :))); % Index 16 is dtheta
% --- NEW: Calculate Max x and Max ddot_x ---
max_cart_pos_excursion = max(abs(x_hist));
max_cart_acceleration = max(abs(ddot_x_hist));
% -------------------------------------------
% Calculate max angle deviation during swing-up (before stabilization starts)
if t_switch_index > 1
    max_angle_during_swingup = max(abs(wrapToPi(x_unified(14, 1:t_switch_index)))); 
else
    max_angle_during_swingup = max(abs(wrapToPi(x_unified(14, :)))); % Use full array if no switch occurred
end
fprintf('\nD. Effort and Peak Values:\n');
fprintf('   - Control Effort J = integral(F_c^2 dt): %10.3f N^2*s\n', J_control_effort);
fprintf('   - Max Angular Velocity max(|dtheta|): %10.3f rad/s\n', max_angular_velocity);
fprintf('   - Max Swing-up Angle (Initial): %10.3f rad (%4.1f deg)\n', max_angle_during_swingup, rad2deg(max_angle_during_swingup));
% --- NEW: Print Statements ---
fprintf('   - Max Cart Excursion max(|x|): %10.3f m\n', max_cart_pos_excursion);
fprintf('   - Max Cart Velocity max(|dx|): %10.3f m/s\n', max_cart_velocity);
fprintf('   - Max Cart Acceleration max(|ddot_x|): %10.3f m/s^2\n', max_cart_acceleration);
% -----------------------------
fprintf('======================================================\n');
% --- End of Performance Analysis ---
%% ================================================================================
%% SECTION 6: RESULTS VISUALIZATION (COMPREHENSIVE FIGURES)
%% ================================================================================
% Color Definitions
COLOR_REF = [0.5 0.5 0.5];      % Grey
COLOR_NM = [0.0, 0.45, 0.7]; % Deep Blue (Unified NM)
COLOR_DIST = [0.9 0.0, 0.0];  % Bright Red for Disturbance
COLOR_STAB = [0.85 0.33 0.1]; % Red/Orange for Stabilization mode
% Helper function for shading
function add_disturb_layer(t_start, t_end, y_lim, COLOR_DIST)
    patch([t_start t_end t_end t_start], [y_lim(1) y_lim(1) y_lim(2) y_lim(2)], COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end
%% --- FIGURE 1: Phase Portrait ($\theta$ vs $\dot{\theta}$) ---
figure('Name', 'Figure 1: Pendulum Phase Portrait (Neuromorphic Car-Pendulum)', 'Position', [100, 100, 800, 220]);
hold on;
plot(theta_hist, omega_hist, 'Color', COLOR_NM, 'LineWidth', 2, 'DisplayName', 'NM Trajectory');
plot(theta_hist(1), omega_hist(1), 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');
t_idx = find(t >= t_disturb_start, 1);
plot(theta_hist(t_idx), omega_hist(t_idx), 'p', 'MarkerSize', 12, 'LineWidth', 2, 'Color', COLOR_DIST, 'DisplayName', 'Disturbance Start');
plot(theta_hist(end), omega_hist(end), 'ks', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'End');
plot(pi, 0, 'rx', 'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', 'Target ($\pi$, Inverted)'); 
plot(0, 0, 'mo', 'MarkerSize', 6, 'LineWidth', 2, 'DisplayName', 'Stable Equil. (0)');
xlabel('Angle $\theta$ (rad)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('Angular Velocity $\dot{\theta}$ (rad/s)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 12);
legend('Location', 'best', 'Interpreter', 'latex');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
%% --- FIGURE 2: TIME HISTORY OF ALL STATE VARIABLES ---
figure('Name', 'Figure 2: Car-Pendulum State Time History', 'Position', [100, 100, 800, 550]);
% --- Subplot 1: Cart Position (x) ---
subplot(4,1,1);
hold on;
ylim([-0.2, 0.8]);   % <-- Already present
y_lim_x = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_x, COLOR_DIST);
plot(t, x_hist, 'Color', '#0090d0', 'LineWidth', 2, 'DisplayName', 'Cart Position $x$'); 
ylabel('Position $x$ (m)', 'Interpreter', 'latex');
grid on;
legend('Disturbance Window', 'Interpreter', 'latex');
yticks([-0.4:0.2:0.8]);
% --- Subplot 2: Pendulum Angle ($\theta$) ---
subplot(4,1,2);
hold on;
ylim([-0.5, 5]);   % <-- ADDED
yline(pi, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Target $\pi$');
plot(t, theta_hist, 'Color', '#edb120', 'LineWidth', 2); 
ylabel('Angle $\theta$ (rad)', 'Interpreter', 'latex');
grid on;
y_lim_theta = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_theta, COLOR_DIST);
legend('Target $\pi$','Interpreter', 'latex');
yticks([0:1:5]);
% --- Subplot 3: Cart Velocity ($\dot{x}$) ---
subplot(4,1,3);
hold on;
ylim([-1.2, 1.2]);   % <-- ADDED
plot(t, dx_hist, 'Color', '#dd5400', 'LineWidth', 2, 'DisplayName', 'Cart Velocity $\dot{x}$'); 
yline(0, 'k:', 'LineWidth', 1);
ylabel('Cart Vel. $\dot{x}$ (m/s)', 'Interpreter', 'latex');
grid on;
y_lim_dx = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_dx, COLOR_DIST);
% --- Subplot 4: Angular Velocity ($\dot{\theta}$) ---
subplot(4,1,4);
hold on;
ylim([-15, 15]);   % <-- ADDED
plot(t, omega_hist, 'Color', '#660099', 'LineWidth', 2, 'DisplayName', 'Ang. Velocity $\dot{\theta}$'); 
yline(0, 'k:', 'LineWidth', 1);
ylabel('Ang. Vel. $\dot{\theta}$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)');
grid on;
y_lim_dtheta = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_dtheta, COLOR_DIST);
%% --- FIGURE 3: CONTROL FORCE AND ENERGY HISTORY ---
figure('Name', 'Figure 3: Control Force and Energy History', 'Position', [100, 150, 800, 410]);
% --- Subplot 1: Total Applied Force (Controller + Disturbance) ---
subplot(2,1,1);
hold on;
plot(t, force_hist, 'Color', COLOR_NM, 'LineWidth', 2, 'DisplayName', 'Total Applied Force $F$'); 
plot(t, disturb_hist, 'Color', COLOR_DIST, 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'Disturbance $F_{ext}$'); 
yline(max_abs_force, 'K--', 'LineWidth', 1.0, 'DisplayName', 'Max Controller Limit');
yline(-max_abs_force, 'K--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
ylabel('Force $F$ (N)', 'Interpreter', 'latex');
legend('show', 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 10);
grid on;
y_lim_F = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_F, COLOR_DIST);
% --- Subplot 2: Total Mechanical Energy (Cart + Pendulum) ---
subplot(2,1,2);
hold on;
plot(t, E_mechanical_hist, 'Color', '#3baa32', 'LineWidth', 2, 'DisplayName', 'Total Mechanical Energy $E$');
yline(E_target, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Pendulum $E_{target}$'); 
ylabel('Total Energy (J)', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
legend('show', 'Location', 'best', 'Interpreter', 'latex');
grid on;
y_lim_E = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_E, COLOR_DIST);
%% --- FIGURE 4: NEUROMORPHIC INTERNAL DYNAMICS (NEW) ---
figure('Name', 'Figure 4: Neuromorphic Internal Dynamics', 'Position', [100, 100, 800, 500]);
% --- Subplot 1: NM Fast States V1a and V2a (Main Opposing Oscillators) ---
subplot(3,1,1);
hold on;
% V1a is x_unified(1, :) and V2a is x_unified(7, :)
plot(t, x_unified(1, :), 'Color', COLOR_NM, 'LineWidth', 2, 'DisplayName', '$V_{1a}$ (Pos. Drive)'); 
plot(t, x_unified(7, :), 'Color', COLOR_STAB, 'LineWidth', 2, 'DisplayName', '$V_{2a}$ (Neg. Drive)'); 
yline(threshold, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Threshold (0)');
ylabel('Voltages in HCO$_{1}$', 'Interpreter', 'latex');
grid on;
y_lim_v1v2 = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_v1v2, COLOR_DIST);
legend('$V_{1}$', '$V_{2}$', 'Threshold (0)', 'Interpreter', 'latex', 'Location', 'best');
% title('HCO Fast States and Raw NM Force Output', 'FontWeight', 'bold');
% --- Subplot 2: NM Fast States V1b and V2b (Cross-coupled Oscillators) ---
subplot(3,1,2);
hold on;
% V1b is x_unified(4, :) and V2b is x_unified(10, :)
plot(t, x_unified(4, :), 'Color', '#edb120', 'LineStyle', '-', 'LineWidth', 2, 'DisplayName', '$V_{3}$'); 
plot(t, x_unified(10, :), 'Color', '#8516d1', 'LineStyle', '-', 'LineWidth', 2, 'DisplayName', '$V_{4}$'); 
yline(threshold, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Threshold (0)');
ylabel('Voltages in HCO$_{2}$', 'Interpreter', 'latex');
legend('show', 'Interpreter', 'latex', 'Location', 'best');
grid on;
y_lim_v3v4 = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_v3v4, COLOR_DIST);
legend('$V_{3}$', '$V_{4}$', 'Threshold (0)', 'Interpreter', 'latex', 'Location', 'best');
% --- Subplot 3: Raw NM Force Output (Before Clamping/Disturbance) ---
subplot(3,1,3);
hold on;
ylim([-5, 5]);   % <-- ADDED
plot(t, disturb_hist, 'Color', COLOR_DIST, 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'Disturbance $F_{ext}$'); 
% add_control_layer(t, control_mode, T_final, y_lim_F);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], [-5 -5 5 5], COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Disturbance Window');
plot(t, raw_nm_force_hist, 'Color', '#3baa32', 'LineWidth', 2, 'DisplayName', 'Raw NM Force $F_{NM}$'); 
yline(max_abs_force, 'K--', 'LineWidth', 1.0, 'DisplayName', 'Force Limit');
yline(-max_abs_force, 'K--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
ylabel('Raw Force $F_{NM}$ (N)', 'Interpreter', 'latex');
xlabel('Time (s)');
legend('show', 'Interpreter', 'latex', 'Location', 'best');
grid on;
%% --- FIGURE 5: CONTROL MODE HISTORY (MOVED FROM FIGURE 4) ---
figure('Name', 'Figure 5: Control Mode Switching', 'Position', [100, 100, 600, 200]);
hold on;
% Plot mode history (scaled for visibility)
plot(t(control_mode_hist == 2), control_mode_hist(control_mode_hist == 2) * 0.5, 'Color', COLOR_STAB, 'LineWidth', 5, 'DisplayName', 'Stabilization (2)');
plot(t(control_mode_hist == 1), control_mode_hist(control_mode_hist == 1) * 0.5, 'Color', COLOR_NM, 'LineWidth', 5, 'DisplayName', 'Swing-up (1)');
    
yline(1.0, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
ylim([0.4 1.1]);
yticks([0.5, 1.0]);
yticklabels({'Swing-up (1)', 'Stabilization (2)'});
ylabel('Control Mode', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
title('Control Mode Switching based on Position Condition');
grid on;
y_lim_mode = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_mode, COLOR_DIST);
fprintf('All figures generated successfully.\n');
