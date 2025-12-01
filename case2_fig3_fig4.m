% This script implements a unified neuromorphic controller for a car-pendulum system.
% It uses Dual Half-Center Oscillators (HCOs) for swing-up and error-based 
% modulation for stabilization.
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
clear; clc; close all; % Clear workspace, command window, and close figures

%% SECTION 1: REFERENCE SIGNAL GENERATION
%% ================================================================================
Tf = 30; % Total simulation time (seconds)
f_theta = 10; % Base frequency parameter for timing scaling
dt = 1/f_theta/1e2; % Simulation time step (0.001s)
t = 0:dt:Tf; % Time vector array
theta_ref = pi * ones(size(t)); % Reference angle trajectory (Target: pi radians/upright)
% Assuming cart position reference is 0 for stabilization
x_ref = zeros(size(t)); % Reference cart position (Target: 0 meters)
theta_interp = @(tq) interp1(t, theta_ref, tq, 'nearest', 'extrap'); % Interpolation function for angle ref
dot_theta_ref = @(tq) 0; % Function handle for reference angular velocity (0 rad/s)

%% ================================================================================
%% SECTION 2: CAR-PENDULUM PHYSICAL PARAMETERS
%% ================================================================================
% --- Cart Parameters ---
Mc = 1.0;           % Mass of the Cart (kg)
bc = 0.5;           % Viscous Damping coefficient on Cart (N-s/m)

% --- Pendulum Link Parameters ---
m = 0.024;          % Mass of Pendulum Link (kg)
L = 0.129;          % Total Length of Pendulum Link (m)
b = 0.00005;        % Viscous Damping coefficient of Pendulum Link (N-m-s/rad)
g = 9.81;           % Gravitational acceleration (m/s^2)
L_half = L/2;       % Distance from pivot to Center of Mass (assuming uniform rod)
Jperp = (1/3) * m * L^2; % Moment of inertia about the pivot point (kg-m^2)

% Target energy for the upright position (E = m * g * h_cm)
E_target = m * g * L; % Potential energy at upright position (relative to stable equilibrium)

% Print physical parameters to console
fprintf('Using Car-Pendulum Dynamics:\n');
fprintf('  Cart Mass (Mc): %.3f kg, Pendulum Mass (m): %.3f kg\n', Mc, m);
fprintf('Target mechanical energy for upright position (theta=pi): %.3f J\n', E_target);

%% ================================================================================
%% SECTION 3: NEUROMORPHIC CONTROLLER PARAMETER TUNING
%% ================================================================================
% --- Neuromorphic Parameters ---
f_base = 1; % Base frequency for the neural oscillator
tau_f_base = 0.1; % Base time constant for fast neurons
tau_s_base = 20*tau_f_base; % Base time constant for slow adaptation
tau_us_base = 100*tau_f_base; % Base time constant for ultra-slow adaptation

% Scaled time constants based on the system frequency
tau_f = tau_f_base * f_base / f_theta; 
tau_s = tau_s_base * f_base / f_theta;
tau_us = tau_us_base * f_base / f_theta;

force_gain = 5; % Gain to convert neural output to force (initially for swing-up)
I_ext_base = -1.0; % Baseline external current to drive intrinsic oscillation
threshold = 0; % Threshold for sigmoid activation function

% HCO Parameters (Neural Connection Weights)
g_f = 2;        % Self-excitation gain
g_splus = 1.5;  % Fast adaptation gain
g_sminus = 1;   % Slow adaptation gain
g_us = 1.5;     % Ultra-slow adaptation gain
a_cross = 1.5;  % Cross-inhibition weight between HCOs

% Synaptic transfer function (Sigmoid)
synapse = @(vs,gain) gain./(1 + exp(-2*(vs+1))); 

% --- Adaptation & Feedback Parameters ---
learning_rate_energy = 1; % Rate at which force gain adapts based on energy error
Ki_gain = 0.01; % Integral gain for energy error adaptation

% Gains for stabilization feedback (applied as FORCE on the cart)
K_error_feedback = 100; % Proportional gain for angle error
K_velocity_feedback = 20; % Derivative gain for angular velocity

% Maximum FORCE limit (Saturation)
max_abs_force = 5; % N (Max force actuator can apply) 

% --- Position-based Stabilization Parameters ---
theta_tolerance = deg2rad(5); % Angle tolerance (+/- 5 deg) to switch to stabilization mode
I_ext_stabilization = 2;     % Positive input current to suppress oscillation during stabilization

fprintf('Maximum controller force applied: %.2f N\n', max_abs_force);
fprintf('Stabilization mode activates when |theta - pi| < %.2f rad (%.1f deg)\n', theta_tolerance, rad2deg(theta_tolerance));

% --- MODIFIED: Sinusoidal Disturbance Parameters (Force on Cart) ---
A_disturb = 1.0;       % Amplitude of the disturbance sine wave (N)
f_disturb = 2.0;       % Frequency of the disturbance sine wave (Hz)
t_disturb_start = Tf/2; % Start time for disturbance (s)
t_disturb_end = Tf;   % End time for disturbance (s)

fprintf('Sinusoidal disturbance (A=%.1f N, f=%.1f Hz) applied between t=%.1f s and t=%.1f s.\n', A_disturb, f_disturb, t_disturb_start, t_disturb_end);

%% ================================================================================
%% SECTION 4: SIMULATION: UNIFIED NEUROMORPHIC CONTROL (POSITION-CONDITIONED)
%% ================================================================================
fprintf('\nStarting Simulation: Car-Pendulum with Neuromorphic Controller...\n');

% State vector initialization: 
% [v1...v12 (Neural states); x (13); theta (14); dx (15); dtheta (16)] = 16 states total
x_unified = zeros(16, length(t));

% Initial conditions: 
% Neural states = 0; Cart x = 0; Pendulum theta = 0.1 rad; Velocities = 0
x_unified(:,1) = [zeros(1, 12), 0, 0.1, 0, 0]; 
x_unified(3,1) = -1; % Perturb neuron 3 (adaptation) to break symmetry and start oscillation
x_unified(9,1) = -1; % Perturb neuron 9 (adaptation) to break symmetry

% Initialize adaptive gains and integral term variables
force_pos_gain = force_gain; % Positive direction force gain
force_neg_gain = force_gain; % Negative direction force gain
energy_error_integral = 0;   % Accumulator for energy error integral
current_I_ext = I_ext_base;  % Set initial external input current

% Initialize logging variables for post-simulation plotting
x_hist = zeros(size(t));      % History of cart position
dx_hist = zeros(size(t));     % History of cart velocity
theta_hist = zeros(size(t));  % History of pendulum angle
omega_hist = zeros(size(t));  % History of angular velocity
force_hist = zeros(size(t));  % History of final applied force
raw_nm_force_hist = zeros(size(t)); % History of raw neural output force
E_mechanical_hist = zeros(size(t)); % History of total energy
control_mode_hist = zeros(size(t)); % History of control mode (1=Swing, 2=Stab)
disturb_hist = zeros(size(t));      % History of applied disturbance
F_controller_history = zeros(size(t)); % History of controller force (pre-disturbance)

% --- NEW: Acceleration History Initialization ---
ddot_x_hist = zeros(size(t)); % History of cart acceleration
% ------------------------------------------------

% Log the initial state into history arrays
x_hist(1) = x_unified(13,1);
theta_hist(1) = x_unified(14,1);
dx_hist(1) = x_unified(15,1);
omega_hist(1) = x_unified(16,1);

% Calculate initial energy for logging
KE_p = 0.5 * Jperp * x_unified(16,1)^2; % Kinetic energy (rotational)
PE_p = m * g * L_half * (1 - cos(x_unified(14,1))); % Potential energy
KE_c = 0.5 * Mc * x_unified(15,1)^2; % Kinetic energy (translational)
E_mechanical_hist(1) = KE_p + PE_p + KE_c; % Total energy
control_mode_hist(1) = 1; % Set initial control mode to Swing-up (1)

% --- MAIN SIMULATION LOOP ---
for k = 2:length(t)
    % Extract current physical states from the unified vector (from previous step)
    x = x_unified(13,k-1);
    theta = x_unified(14,k-1);
    dx = x_unified(15,k-1);
    omega_p = x_unified(16,k-1); % dtheta
    
    % Calculate angular error relative to the target inverted position (pi)
    theta_error = wrapToPi(theta_ref(k-1) - theta);
    dot_theta_error = dot_theta_ref(t(k-1)) - omega_p;
    
    % --- Energy Calculation (Pendulum-Only Mechanical Energy) ---
    KE_p = 0.5 * Jperp * omega_p^2; % Rotational Kinetic Energy
    PE_p = m * g * L_half * (1 - cos(theta)); % Potential Energy (0 at bottom)
    E_current = KE_p + PE_p; % Total Pendulum Energy
    energy_error = E_target - E_current; % Deviation from upright energy
    
    % --- Position-Based Switching Condition ---
    % Check if pendulum is within tolerance of the upright position
    is_stabilizing = abs(theta_error) < theta_tolerance;
    
    % --- Adaptive Gain & I_ext Logic ---
    if is_stabilizing 
        % MODE 2: Stabilization Phase
        current_I_ext = I_ext_stabilization; % Set input to positive to stop oscillation
        
        % Fix gains to a low constant value for fine control
        force_pos_gain = 1; 
        force_neg_gain = 1; 
        gain_update = 0; % No adaptive learning in this phase
        
        control_mode_hist(k) = 2; % Log Stabilization mode
    else
        % MODE 1: Swing-up Phase (Energy Pumping)
        current_I_ext = I_ext_base; % Set input negative to enable oscillation
        
        % Calculate adaptive gain update based on energy error
        energy_error_integral = energy_error_integral + energy_error * dt;
        gain_update = learning_rate_energy * energy_error + Ki_gain * energy_error_integral;
        
        % Update force gains (asymmetric update logic)
        force_pos_gain = force_pos_gain + gain_update * dt;
        force_neg_gain = force_neg_gain - gain_update * dt;
        
        % Clamp gains to ensure stability and positive values
        max_adaptive_gain = 5; 
        force_pos_gain = max(min(force_pos_gain, max_adaptive_gain), 1e-4);
        force_neg_gain = max(min(force_neg_gain, max_adaptive_gain), 1e-4);
        
        control_mode_hist(k) = 1; % Log Swing-up mode
    end
    
    % --- 1. Neuromorphic Force Calculation (Always Active) ---
    % Calculate sigmoid outputs of the two opposing motor neurons (V1 and V7)
    hco1_output = (1./(1 + exp(-2*(x_unified(1, k-1) - threshold))));
    hco2_output = (1./(1 + exp(-2*(x_unified(7, k-1) - threshold))));
    
    % Calculate raw NM output as difference of weighted outputs
    force_nm = force_pos_gain * hco1_output - force_neg_gain * hco2_output;
    raw_nm_force_hist(k) = force_nm; % Log raw force output
    
    F_controller = force_nm; % Assign to controller variable
    
    % --- 2. Unified External Input Modulation (The Stabilization) ---
    % Calculate stabilization feedback (PD controller on theta error)
    error_feedback = K_error_feedback * theta_error + K_velocity_feedback * dot_theta_error;
    
    % Modulate external neural inputs: Add feedback to drive one side or the other
    input_hco1 = current_I_ext + max(error_feedback, 0); 
    input_hco2 = current_I_ext + max(-error_feedback, 0); 
    
    % Saturate (Clamp) controller Force to physical limits
    F_controller = max(min(F_controller, max_abs_force), -max_abs_force);
    
    % --- Log F_controller (before disturbance is added) ---
    F_controller_history(k) = F_controller;
    
    % --- MODIFIED: External Sinusoidal Disturbance Logic ---
    current_disturb = 0;
    if t(k-1) >= t_disturb_start && t(k-1) < t_disturb_end
        % Calculate sine wave disturbance value
        current_disturb = A_disturb * sin(2 * pi * f_disturb * (t(k-1) - t_disturb_start));
    end
    
    % Calculate final Force applied to the system (Controller + Disturbance)
    applied_force = F_controller + current_disturb;
    
    % --- DUAL HCO NEUROMORPHIC DYNAMICS (Calculated for t[k-1]) ---
    dxdt = zeros(16,1);
    
    % HCO Group 1 (Neurons 1-6) - Positive Force Drive
    % Neuron 1 (Membrane Voltage):
    dxdt(1) = (-x_unified(1,k-1) + g_f*tanh(x_unified(1,k-1)) - g_splus*tanh(x_unified(2,k-1)) + g_sminus*tanh(x_unified(2,k-1)+0.9) - g_us*tanh(x_unified(3,k-1)+0.9) + synapse(x_unified(5,k-1), -0.2) - a_cross*tanh(x_unified(7,k-1)) + input_hco1) / tau_f;
    dxdt(2) = (x_unified(1,k-1) - x_unified(2,k-1)) / tau_s; % Adaptation variable 1
    dxdt(3) = (x_unified(1,k-1) - x_unified(3,k-1)) / tau_us; % Ultra-slow adaptation 1
    % Neuron 2 (Interneuron/Auxiliary):
    dxdt(4) = (-x_unified(4,k-1) + g_f*tanh(x_unified(4,k-1)) - g_splus*tanh(x_unified(5,k-1)) + g_sminus*tanh(x_unified(5,k-1)+0.9) - g_us*tanh(x_unified(6,k-1)+0.9) + synapse(x_unified(2,k-1), -0.2) + a_cross*tanh(x_unified(7,k-1)) + input_hco1) / tau_f;
    dxdt(5) = (x_unified(4,k-1) - x_unified(5,k-1)) / tau_s; % Adaptation variable 2
    dxdt(6) = (x_unified(4,k-1) - x_unified(6,k-1)) / tau_us; % Ultra-slow adaptation 2
    
    % HCO Group 2 (Neurons 7-12) - Negative Force Drive
    % Neuron 3 (Membrane Voltage):
    dxdt(7) = (-x_unified(7,k-1) + g_f*tanh(x_unified(7,k-1)) - g_splus*tanh(x_unified(8,k-1)) + g_sminus*tanh(x_unified(8,k-1)+0.9) - g_us*tanh(x_unified(9,k-1)+0.9) + synapse(x_unified(11,k-1), -0.2) - a_cross*tanh(x_unified(1,k-1)) + input_hco2) / tau_f;
    dxdt(8) = (x_unified(7,k-1) - x_unified(8,k-1)) / tau_s; % Adaptation variable 3
    dxdt(9) = (x_unified(7,k-1) - x_unified(9,k-1)) / tau_us; % Ultra-slow adaptation 3
    % Neuron 4 (Interneuron/Auxiliary):
    dxdt(10) = (-x_unified(10,k-1) + g_f*tanh(x_unified(10,k-1)) - g_splus*tanh(x_unified(11,k-1)) + g_sminus*tanh(x_unified(11,k-1)+0.9) - g_us*tanh(x_unified(12,k-1)+0.9) + synapse(x_unified(8,k-1), -0.2) + a_cross*tanh(x_unified(1,k-1)) + input_hco2) / tau_f;
    dxdt(11) = (x_unified(10,k-1) - x_unified(11,k-1)) / tau_s; % Adaptation variable 4
    dxdt(12) = (x_unified(10,k-1) - x_unified(12,k-1)) / tau_us; % Ultra-slow adaptation 4
    
    % --- CAR-PENDULUM COUPLED DYNAMICS ---
    % State variables: [x (13); theta (14); dx (15); dtheta (16)]
    
    % Inertia of pendulum about pivot
    I_p = Jperp;
    
    % Solving Linear System A * ddot(X) = B derived from Euler-Lagrange equations
    % Matrix A elements:
    A11 = Mc + m;
    A12 = m * L_half * cos(theta);
    % Vector B elements (including Coriolis, Gravity, Damping, External Force):
    B1  = applied_force - bc * dx + m * L_half * omega_p^2 * sin(theta);
    
    A21 = m * L_half * cos(theta);
    A22 = I_p + m * L_half^2;
    B2  = m * g * L_half * sin(theta) - b * omega_p; 
    
    % Determinant of Matrix A
    Delta = A11 * A22 - A12 * A21;
    
    % Solve for accelerations using Cramer's rule
    ddot_x = (B1 * A22 - B2 * A12) / Delta; % Acceleration of Cart
    ddot_theta = (A11 * B2 - A21 * B1) / Delta; % Angular Acceleration of Pendulum
    
    % --- NEW: Store Acceleration for Printing ---
    ddot_x_hist(k) = ddot_x; % Log cart acceleration
    % --------------------------------------------

    % Assemble derivative vector for physical states
    dxdt(13) = dx;              % x_dot
    dxdt(14) = omega_p;         % theta_dot
    dxdt(15) = ddot_x;          % x_ddot
    dxdt(16) = ddot_theta;      % theta_ddot
    
    % --- State Update (Euler Integration) ---
    x_unified(:,k) = x_unified(:,k-1) + dt * dxdt;
    
    % --- CONSTRAINT: Angle Wrapping (Optional/Commented out for continuous tracking) ---
    % x_unified(14,k) = wrapToPi(x_unified(14,k)); 
    x_unified(14,k) = (x_unified(14,k)); 
    
    % --- Data Logging ---
    x_hist(k) = x_unified(13,k);      % Store position
    theta_hist(k) = x_unified(14,k);  % Store angle
    dx_hist(k) = x_unified(15,k);     % Store velocity
    omega_hist(k) = x_unified(16,k);  % Store angular velocity
    force_hist(k) = applied_force;    % Store applied force
    disturb_hist(k) = current_disturb; % Store disturbance value
    
    % Log total mechanical energy (Cart + Pendulum)
    KE_p_log = 0.5 * Jperp * x_unified(16,k)^2;
    PE_p_log = m * g * L_half * (1 - cos(x_unified(14,k)));
    KE_c_log = 0.5 * Mc * x_unified(15,k)^2;
    E_mechanical_hist(k) = KE_p_log + PE_p_log + KE_c_log;
end

% Local Helper function to wrap angle to [-pi, pi]
function angle_wrapped = wrapToPi(angle)
    angle_wrapped = angle - 2*pi*floor((angle + pi)/(2*pi));
end

%% ================================================================================
%% SECTION 5: STATISTICAL ANALYSIS AND PRINTING
%% ================================================================================
% --- 5.1 Pre-Calculation of Errors and Indices ---
% Calculate errors against reference signals (theta_ref = pi, x_ref = 0)
theta_error_full = wrapToPi(theta_ref - theta_hist);
x_error_full = x_ref - x_hist; 

% Find indices for specific time intervals for analysis
t_13s_idx = find(t >= Tf/2-2, 1, 'first');
t_15s_idx = find(t <= Tf/2, 1, 'last');
t_28s_idx = find(t >= Tf-2, 1, 'first');
t_30s_idx = find(t <= Tf, 1, 'last');

% Find the index where the controller first switches to stabilization mode (mode 2)
t_switch_index = find(control_mode_hist == 2, 1, 'first');

% --- 5.2 Overall Performance Metrics (First 15s) ---
max_cart_p = max( x_hist); % Find maximum cart position
max_force = max(abs(force_hist)); % Find maximum applied force magnitude
% Calculate Total Work Done on Cart: integral(|F| * |dx|) dt
force_work_integral = sum(abs(force_hist) .* abs(dx_hist)) * dt; 

fprintf('\n--- Performance Statistics (Neuromorphic Car-Pendulum Control) ---\n');
fprintf('A. Overall Metrics (Tf = %.1f s):\n', Tf);
fprintf('   1. Maximum Applied Force: %.3f m', max_cart_p); % Note: Label says Force, val is Position?
fprintf('   1. Maximum Cart Position: %.3f N\n', max_force); % Note: Label says Position, val is Force?
fprintf('   2. Total Work Done on Cart: %.2f J\n', force_work_integral);
fprintf('   3. Final Cart Position: %.3f m\n', x_hist(end));
fprintf('-------------------------------------------------------------\n');

% --- 5.3 Steady-State Performance (Pre-Disturbance Interval) ---
idx_ss1 = t_13s_idx:t_15s_idx;
theta_err_ss1 = theta_error_full(idx_ss1);
x_err_ss1 = x_error_full(idx_ss1);
theta_max_err_ss1 = max(abs(theta_err_ss1))/pi*100; % Percentage max error
theta_rms_err_ss1 = rms(theta_err_ss1)/pi*100;      % Percentage RMS error
x_max_err_ss1 = max(abs(x_err_ss1));
x_rms_err_ss1 = rms(x_err_ss1);

fprintf('\nB. Steady-State Performance (Tf/2-2 s - Tf/2 s, Pre-Disturbance):\n');
fprintf('   | State | Max Error | RMS Error |\n');
fprintf('   |-------|-----------|-----------|\n');
fprintf('   | Angle | %9.5f na | %9.5f na |\n', theta_max_err_ss1, theta_rms_err_ss1);
fprintf('   | Cart X| %9.5f m   | %9.5f m   |\n', x_max_err_ss1, x_rms_err_ss1);

% --- 5.4 Disturbance Rejection Performance (Post-Disturbance Interval) ---
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
% Control Efficiency: Integral of squared control force (Energy Cost)
J_control_effort = sum(F_controller_history.^2) * dt; 
max_cart_velocity = max(abs(x_unified(15, :))); % Max abs velocity of cart
max_angular_velocity = max(abs(x_unified(16, :))); % Max abs angular velocity of pendulum

% --- NEW: Calculate Max x and Max ddot_x ---
max_cart_pos_excursion = max(abs(x_hist)); % Max absolute cart position
max_cart_acceleration = max(abs(ddot_x_hist)); % Max absolute cart acceleration
% -------------------------------------------

% Calculate max angle deviation during swing-up (before stabilization starts)
if t_switch_index > 1
    max_angle_during_swingup = max(abs(wrapToPi(x_unified(14, 1:t_switch_index)))); 
else
    max_angle_during_swingup = max(abs(wrapToPi(x_unified(14, :)))); % Use full array if no switch
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
% Color Definitions for plotting
COLOR_REF = [0.5 0.5 0.5];      % Grey color code
COLOR_NM = [0.0, 0.45, 0.7];    % Deep Blue (Unified NM) color code
COLOR_DIST = [0.9 0.0, 0.0];    % Bright Red for Disturbance color code
COLOR_STAB = [0.85 0.33 0.1];   % Red/Orange for Stabilization mode color code

% Helper function for adding shaded regions for disturbance
function add_disturb_layer(t_start, t_end, y_lim, COLOR_DIST)
    patch([t_start t_end t_end t_start], [y_lim(1) y_lim(1) y_lim(2) y_lim(2)], COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end

%% --- FIGURE 1: Phase Portrait ($\theta$ vs $\dot{\theta}$) ---
figure('Name', 'Figure 1: Pendulum Phase Portrait (Neuromorphic Car-Pendulum)', 'Position', [100, 100, 800, 220]);
hold on;
plot(theta_hist, omega_hist, 'Color', COLOR_NM, 'LineWidth', 2, 'DisplayName', 'NM Trajectory'); % Trajectory
plot(theta_hist(1), omega_hist(1), 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start'); % Start point
t_idx = find(t >= t_disturb_start, 1); % Find index where disturbance starts
plot(theta_hist(t_idx), omega_hist(t_idx), 'p', 'MarkerSize', 12, 'LineWidth', 2, 'Color', COLOR_DIST, 'DisplayName', 'Disturbance Start'); % Disturbance point
plot(theta_hist(end), omega_hist(end), 'ks', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'End'); % End point
plot(pi, 0, 'rx', 'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', 'Target ($\pi$, Inverted)'); % Target point
plot(0, 0, 'mo', 'MarkerSize', 6, 'LineWidth', 2, 'DisplayName', 'Stable Equil. (0)'); % Stable equilibrium
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
ylim([-0.2, 0.8]);   % Set Y-limits for Cart Position
y_lim_x = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_x, COLOR_DIST); % Add shaded region
plot(t, x_hist, 'Color', '#0090d0', 'LineWidth', 2, 'DisplayName', 'Cart Position $x$'); 
ylabel('Position $x$ (m)', 'Interpreter', 'latex');
grid on;
legend('Disturbance Window', 'Interpreter', 'latex');
yticks([-0.4:0.2:0.8]);

% --- Subplot 2: Pendulum Angle ($\theta$) ---
subplot(4,1,2);
hold on;
ylim([-0.5, 5]);   % Set Y-limits for Angle
yline(pi, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Target $\pi$'); % Target Line
plot(t, theta_hist, 'Color', '#edb120', 'LineWidth', 2); 
ylabel('Angle $\theta$ (rad)', 'Interpreter', 'latex');
grid on;
y_lim_theta = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_theta, COLOR_DIST); % Add shaded region
legend('Target $\pi$','Interpreter', 'latex');
yticks([0:1:5]);

% --- Subplot 3: Cart Velocity ($\dot{x}$) ---
subplot(4,1,3);
hold on;
ylim([-1.2, 1.2]);   % Set Y-limits for Cart Velocity
plot(t, dx_hist, 'Color', '#dd5400', 'LineWidth', 2, 'DisplayName', 'Cart Velocity $\dot{x}$'); 
yline(0, 'k:', 'LineWidth', 1); % Zero line
ylabel('Cart Vel. $\dot{x}$ (m/s)', 'Interpreter', 'latex');
grid on;
y_lim_dx = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_dx, COLOR_DIST); % Add shaded region

% --- Subplot 4: Angular Velocity ($\dot{\theta}$) ---
subplot(4,1,4);
hold on;
ylim([-15, 15]);   % Set Y-limits for Angular Velocity
plot(t, omega_hist, 'Color', '#660099', 'LineWidth', 2, 'DisplayName', 'Ang. Velocity $\dot{\theta}$'); 
yline(0, 'k:', 'LineWidth', 1); % Zero line
ylabel('Ang. Vel. $\dot{\theta}$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)');
grid on;
y_lim_dtheta = get(gca, 'YLim');
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_dtheta, COLOR_DIST); % Add shaded region

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
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_F, COLOR_DIST); % Add shaded region

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
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_E, COLOR_DIST); % Add shaded region

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
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_v1v2, COLOR_DIST); % Add shaded region
legend('$V_{1}$', '$V_{2}$', 'Threshold (0)', 'Interpreter', 'latex', 'Location', 'best');

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
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_v3v4, COLOR_DIST); % Add shaded region
legend('$V_{3}$', '$V_{4}$', 'Threshold (0)', 'Interpreter', 'latex', 'Location', 'best');

% --- Subplot 3: Raw NM Force Output (Before Clamping/Disturbance) ---
subplot(3,1,3);
hold on;
ylim([-5, 5]);   % Set Y-limits
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
add_disturb_layer(t_disturb_start, t_disturb_end, y_lim_mode, COLOR_DIST); % Add shaded region
fprintf('All figures generated successfully.\n');
