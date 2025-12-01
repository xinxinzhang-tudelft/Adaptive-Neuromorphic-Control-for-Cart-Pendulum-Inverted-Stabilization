% This script simulates a car-pendulum system using a dual-mode control strategy:
% 1. Energy-based Swing-up Control (Bang-Bang) to bring the pendulum to the upright position.
% 2. PD Stabilization Control to maintain the upright position under disturbances.
%
% The control output is a HORIZONTAL FORCE (F) applied to the cart.
%
% Author: Xinxin Zhang
% Date: 2025-12-01

clear; clc; 
close all; % Clear workspace, command window, and close all figures

%% 1. System Parameters (Car-Pendulum Model) 🏎️
% The system consists of a pendulum attached to a movable cart.
% State vector: X = [x; theta; dx; dtheta]
% x: Cart position (m)
% theta: Pendulum angle (rad), where 0 is downward, pi is upright
% dx: Cart velocity (m/s)
% dtheta: Angular velocity (rad/s)

% --- Cart Parameters ---
Mc = 1.0;           % Mass of the Cart (kg)
bc = 0.5;           % Viscous Damping coefficient on Cart (N-s/m)

% --- Pendulum Link Parameters ---
Mp = 0.024;         % Mass of Pendulum Link (kg) - 'm'
Lp = 0.129;         % Total Length of Pendulum Link (m) - 'L'
Dp = 0.00005;       % Viscous Damping of Pendulum Link (N-s/rad) - 'b'
g = 9.81;           % Gravitational acceleration (m/s^2)

% --- Derived Parameters ---
m = Mp;             % Alias for Pendulum Mass
L = Lp;             % Alias for Pendulum Length
b = Dp;             % Alias for Pendulum Damping
b_cart = bc;        % Alias for Cart Damping
L_half = L/2;       % Distance from pivot to Center of Mass (assuming uniform rod)

% Moment of inertia of the pendulum link about the *pivot* point
% J = I_cm + m*r^2 = (1/12)mL^2 + m(L/2)^2 = (1/3)mL^2
J = (1/3) * m * L^2;

% Target mechanical energy for the upright inverted position (theta = pi)
% Reference potential energy is 0 at the stable downward position (theta = 0)
E_up = m * g * L; % Potential energy at the top = m*g*(2*L_half)

% Maximum allowed control FORCE (Saturation limit for the actuator)
F_max = 5; % N 

% Display parameters to the console
fprintf('Using Car-Pendulum Parameters:\n');
fprintf('  Cart Mass (Mc): %.4f kg, Pendulum Mass (m): %.4f kg\n', Mc, m);
fprintf('  Pendulum Length (L): %.3f m, Inertia (J): %.5f kg*m^2\n', L, J);
fprintf('  Max Control Force (F_max): %.1f N\n', F_max);
fprintf('Target energy for inverted position (E_up): %.3f J\n', E_up);

%% 2. PD Stabilization Controller Design
% These gains are used when the pendulum is near the upright position
Kp = 5; % Proportional Gain (applied to angle error)
Kd = 0.5;  % Derivative Gain (applied to angular velocity)
K_PD = [Kp, Kd]; % Store gains in a vector
disp(['PD Stabilization Gains K = [Kp, Kd] = [', num2str(K_PD(1)), ', ', num2str(K_PD(2)), ']']);

% --- External Disturbance Parameters ---
% Parameters for a sinusoidal force disturbance applied to the cart
T_final = 30;    % Final simulation time (s)
t_disturb_start = T_final/2; % Start disturbance at half-time (15s)
t_disturb_end = T_final;   % End disturbance at final time (30s)
F_disturb_amplitude = 1.0; % Amplitude of disturbance force (N)
f_disturb = 2.0;           % Frequency of disturbance (Hz)

%% 3. Simulation Setup and Control Loop ⚙️
dt = 0.01;      % Time step for integration (s)
t = 0:dt:T_final; % Time vector array
N = length(t);    % Number of simulation steps

% State vector initialization: [x; theta; dx; dtheta] 
X = zeros(4, N); 
X(:, 1) = [0; 0.1; 0; 0]; % Initial conditions: Cart at 0, Pendulum slightly displaced (0.1 rad), zero velocities

% Initialize arrays for logging data
F_history = zeros(1, N); % Log of total applied force (Controller + Disturbance)
F_controller_history = zeros(1, N); % Log of controller output force only
E_history = zeros(1, N); % Log of system total mechanical energy
control_mode = zeros(1, N); % Log of active control mode (1 = Swing-up, 2 = PD Stabilization)

% --- NEW: Acceleration History Initialization ---
ddot_x_history = zeros(1, N); % Log of cart acceleration
% ------------------------------------------------

t_switch = T_final; % Variable to store the time when switching to stabilization occurs
has_switched = false; % Flag to track if switching has happened

% Thresholds for switching condition (from Swing-up to Stabilization)
angle_threshold = deg2rad(30);  % Max angle error to allow switching (+/- 30 degrees)
velocity_threshold = 50.0;     % Max angular velocity to allow switching (rad/s)
energy_tolerance = 0.15 * E_up; % Max energy error to allow switching (15% of target)
CHATTER_DEADBAND = 0.05; % Angular velocity deadband to prevent chattering in Bang-Bang control

% --- FREQUENCY ADJUSTMENT PARAMETER ---
n_sample = 1; % Update Bang-Bang control every n*dt seconds (Decimation factor)
held_force = 0; % Variable to store the force value between updates

disp('Starting Car-Pendulum Energy Control Simulation...');

% --- Main Simulation Loop ---
for k = 1:N-1
    current_time = t(k);
    % Unpack current state
    x = X(1, k);
    theta = X(2, k);
    dx = X(3, k);
    dtheta = X(4, k);
    
    % --- Calculate Current Energy (E) (Pendulum Mechanical Energy) ---
    % Potential Energy (relative to bottom position)
    PE = m * g * L_half * (1 - cos(theta)); 
    % Kinetic Energy (Rotational)
    KE = 0.5 * J * dtheta^2;
    E = PE + KE; % Total Energy
    E_history(k) = E; % Store for plotting
    
    E_error = E_up - E; % Calculate energy error relative to upright target
    
    % Check if state is near the upright equilibrium
    theta_error_from_pi = abs(wrapToPi(theta - pi)); % Angular distance from pi
    is_near_upright = theta_error_from_pi < angle_threshold;
    is_slow_enough = abs(dtheta) < velocity_threshold;
    is_energy_close = abs(E_error) < energy_tolerance;
    
    % --- Determine Control Mode ---
    if has_switched
        control_mode(k) = 2; % Stay in Stabilization Mode
    else
        control_mode(k) = 1; % Stay in Swing-up Mode
    end
    
    % Phase 2: Switch to PD STABILIZATION if conditions are met
    if ~has_switched && is_energy_close && (is_near_upright || is_slow_enough)
        has_switched = true;
        t_switch = current_time;
        control_mode(k) = 2;
        disp(['t = ', num2str(t_switch, '%.3f'), 's: Switching to PD Stabilization']);
    end
    
    % --- Controller FORCE Calculation ---
    F_controller = 0;
    
    if has_switched
        % MODE 2: PD Stabilization
        % Control law: F = -Kp * (theta - pi) - Kd * dtheta
        theta_err = (theta - pi); % Error relative to upright
        F_controller = -K_PD(1) * theta_err - K_PD(2) * dtheta; 
        
    else
        % MODE 1: ENERGY SWING-UP (Bang-Bang Control)
        
        % Only update the bang-bang decision every n_sample steps to simulate discrete control
        if mod(k, n_sample) == 0
            % Energy pumping logic: 
            % To increase energy, force should be applied such that it accelerates the pendulum.
            % The sign depends on angular velocity (dtheta), position (cos theta), and energy error.
            F_sign = sign(dtheta * cos(theta) * E_error);
            
            F_controller = F_max * F_sign; % Apply max force in calculated direction
            
            % Deadband: Set force to 0 if velocity is very small to avoid chattering
            if abs(dtheta) < CHATTER_DEADBAND
                F_controller = 0;
            end
            
            % Save this force to hold it for the next n_sample steps
            held_force = F_controller;
        else
            % Hold the previous force value
            F_controller = held_force;
        end
    end
    
    % Saturate the controller FORCE to physical limits [-F_max, F_max]
    F_controller = max(-F_max, min(F_controller, F_max));
    
    % Log controller force (before disturbance addition)
    F_controller_history(k) = F_controller;
    
    % --- External Disturbance Logic ---
    current_disturb_F = 0;
    if current_time >= t_disturb_start && current_time < t_disturb_end
        % Calculate sinusoidal disturbance based on time elapsed since start
        time_since_start = current_time - t_disturb_start;
        current_disturb_F = F_disturb_amplitude * sin(2 * pi * f_disturb * time_since_start);
    end
    
    % Final FORCE applied to the cart (Control + Disturbance)
    applied_force = F_controller + current_disturb_F;
    
    % Logging total applied force
    F_history(k) = applied_force;
    
    % --- Dynamics (Forward Euler Integration) ---
    I_p = J; % Inertia
    
    % Solving the linear system A * ddot(X) = B for the coupled dynamics
    % Matrix A elements (Inertia Matrix):
    A11 = Mc + m;
    A12 = m * L_half * cos(theta);
    % Vector B elements (Forces/Torques):
    % B1 includes Applied Force, Cart Damping, and Centrifugal Force from pendulum
    B1  = applied_force - b_cart * dx + m * L_half * dtheta^2 * sin(theta);
    
    A21 = m * L_half * cos(theta);
    A22 = I_p + m * L_half^2;
    % B2 includes Gravity and Pendulum Damping
    B2  = m * g * L_half * sin(theta) - b * dtheta; 
    
    % Determinant of Matrix A
    Delta = A11 * A22 - A12 * A21;
    
    % Solve for accelerations using Cramer's rule
    ddot_x = (B1 * A22 - B2 * A12) / Delta; % Linear acceleration of cart
    ddot_theta = (A11 * B2 - A21 * B1) / Delta; % Angular acceleration of pendulum
    
    % --- NEW: Store Acceleration ---
    ddot_x_history(k) = ddot_x; % Log cart acceleration
    % -------------------------------
    
    % Update state using Forward Euler method: x[k+1] = x[k] + dx/dt * dt
    X(1, k+1) = X(1, k) + dx * dt;          % x_next
    X(2, k+1) = X(2, k) + dtheta * dt;      % theta_next
    X(3, k+1) = X(3, k) + ddot_x * dt;      % dx_next
    X(4, k+1) = X(4, k) + ddot_theta * dt;  % dtheta_next
    
    % --- CONSTRAINT: Angle Wrapping is commented out for error calculation consistency
    % X(2, k+1) = wrapToPi(X(2, k+1)); 
    X(2, k+1) = (X(2, k+1)); 
end

% Fill the last element of history arrays to match vector length N
F_controller_history(N) = F_controller_history(N-1);
F_history(N) = F_history(N-1);
E_history(N) = E_history(N-1);
control_mode(N) = control_mode(N-1);
ddot_x_history(N) = ddot_x_history(N-1); % Pad last value

% Display final results
disp(' ');
disp('=== Simulation Complete (Car-Pendulum) ===');
disp(['Final Cart Position: ', num2str(X(1,end), '%.3f'), ' m']);
disp(['Final angle (Unwrapped): ', num2str(rad2deg(X(2,end)), '%.2f'), ' deg']);

%% 5. Performance Analysis and Metrics 📈
% Note: Steady-state is defined after swing-up has completed, which is when 
% control_mode switches to 2 (PD control). The target state is [x=0, theta=pi].
fprintf('\n======================================================\n');
fprintf('  CONTROL PERFORMANCE METRICS \n');
fprintf('======================================================\n');

% Calculate Time Indices for specific intervals
dt = t(2) - t(1);
t_13s = find(t >= 13.0, 1, 'first');
t_15s = find(t <= 15.0, 1, 'last');
t_28s = find(t >= 28.0, 1, 'first');
t_30s = find(t <= 30.0, 1, 'last');

% --- 5.1 System Control Parameters ---
t_switch_index = find(control_mode == 2, 1, 'first');
if isempty(t_switch_index)
    t_switch_actual = NaN; % Did not switch
else
    t_switch_actual = t(t_switch_index); % Actual switching time
end
fprintf('A. Controller Parameters:\n');
fprintf('   - PD Gains [Kp, Kd]:  [%5.2f, %5.2f]\n', Kp, Kd);
fprintf('   - Max Control Force F_max: %5.1f N\n', F_max);
fprintf('   - Actual Switching Time t_switch: %5.3f s\n', t_switch_actual);

% --- 5.2 Error Calculation ---
% Target angle is pi (inverted). Error must be wrapped to [-pi, pi].
theta_error = wrapToPi(X(2, :) - pi); 
% Target cart position is assumed to be 0 for steady-state analysis.
x_error = X(1, :); 

% --- 5.3 Performance Metrics (13s - 15s: Pre-Disturbance SS) ---
idx_ss1 = t_13s:t_15s;
theta_err_ss1 = theta_error(idx_ss1);
x_err_ss1 = x_error(idx_ss1);
theta_max_err_ss1 = max(abs(theta_err_ss1))/pi*100; % Max % Error
theta_rms_err_ss1 = rms(theta_err_ss1)/pi*100;      % RMS % Error
x_max_err_ss1 = max(abs(x_err_ss1));
x_rms_err_ss1 = rms(x_err_ss1);

fprintf('\nB. Steady-State Performance (13s - 15s, Pre-Disturbance):\n');
fprintf('   | State | Max Error | RMS Error |\n');
fprintf('   |-------|-----------|-----------|\n');
fprintf('   | Angle | %9.5f na   | %9.5f  na  |\n', theta_max_err_ss1, theta_rms_err_ss1);
fprintf('   | Cart X| %9.5f m   | %9.5f  m |\n', x_max_err_ss1, x_rms_err_ss1);

% --- 5.4 Performance Metrics (28s - 30s: Post-Disturbance SS) ---
idx_ss2 = t_28s:t_30s;
theta_err_ss2 = theta_error(idx_ss2);
x_err_ss2 = x_error(idx_ss2);
theta_max_err_ss2 = max(abs(theta_err_ss2))/pi*100;
theta_rms_err_ss2 = rms(theta_err_ss2)/pi*100;
x_max_err_ss2 = max(abs(x_err_ss2));
x_rms_err_ss2 = rms(x_err_ss2);

fprintf('\nC. Disturbance Rejection Performance (28s - 30s, Post-Disturbance):\n');
fprintf('   | State | Max Error | RMS Error |\n');
fprintf('   |-------|-----------|-----------|\n');
fprintf('   | Angle | %9.5f na | %9.5f  na |\n', theta_max_err_ss2, theta_rms_err_ss2);
fprintf('   | Cart X| %9.5f m   | %9.5f m   |\n', x_max_err_ss2, x_rms_err_ss2);

% --- 5.5 Control Effort and Peak Values ---
% Control Efficiency: Integral of squared control force
J_control_effort = sum(F_controller_history.^2) * dt; 
max_cart_velocity = max(abs(X(3, :))); 
max_angular_velocity = max(abs(X(4, :)));
max_angle_during_swingup = max(abs(wrapToPi(X(2, 1:t_switch_index)))); 

% --- NEW: Calculate Max x and Max ddot_x ---
max_cart_pos_excursion = max(abs(X(1, :)));
max_cart_acceleration = max(abs(ddot_x_history));
% -------------------------------------------

fprintf('\nD. Effort and Peak Values:\n');
fprintf('   - Control Effort J = integral(F_c^2 dt): %10.3f N^2*s\n', J_control_effort);
fprintf('   - Max Cart Velocity max(|dx|): %10.3f m/s\n', max_cart_velocity);
fprintf('   - Max Angular Velocity max(|dtheta|): %10.3f rad/s\n', max_angular_velocity);
fprintf('   - Max Swing-up Angle (Initial): %10.3f rad (%4.1f deg)\n', max_angle_during_swingup, rad2deg(max_angle_during_swingup));

% --- NEW: Print Statements ---
fprintf('   - Max Cart Excursion max(|x|): %10.3f m\n', max_cart_pos_excursion);
fprintf('   - Max Cart Acceleration max(|ddot_x|): %10.3f m/s^2\n', max_cart_acceleration);
% -----------------------------

fprintf('======================================================\n');
% --- End of Performance Analysis ---

%% 6. Visualization 📊 (Renamed from Section 4)
% Helper function to add control mode switch line
function add_control_layer(t, control_mode, T_final, y_lim)
    mode_changes = find(diff(control_mode) ~= 0);
    if ~isempty(mode_changes)
        t_switch = t(mode_changes(1));
        % Plot vertical dashed line at switch time
        plot([t_switch, t_switch], y_lim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
    end
end
COLOR_DIST = [0.9 0.0, 0.0]; % Red color code for Disturbance

% ... (Visualization figures remain the same structure)

%% --- Figure 1: Pendulum Phase Portrait (PENDULUM ONLY) ---
figure('Name', 'Figure 1: Pendulum Phase Portrait (theta vs dtheta)', 'Position', [100, 100, 800, 200]);
hold on;
energy_idx = (control_mode == 1);
pd_idx = (control_mode == 2);
plot(X(2, energy_idx), X(4, energy_idx), 'r-', 'LineWidth', 2, 'DisplayName', 'Energy Control'); % Phase 1
plot(X(2, pd_idx), X(4, pd_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'PD Control'); % Phase 2
plot(pi, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Target $\pi$ (Inverted)'); % Target
plot(0, 0, 'mo', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Stable Equil.'); % Start
xlabel('Angle $\theta$ (rad)', 'Interpreter', 'latex');
ylabel('Angular Velocity $\dot{\theta}$ (rad/s)', 'Interpreter', 'latex');
legend('Location', 'best');
grid on;
xlim([-pi*1.1, pi*1.1]); 

%% --- Figure 2: Time History of States ---
figure('Name', 'Figure 2: Cart-Pendulum State Time History', ...
       'Position', [750, 100, 800, 550]);
% Global font settings for consistency
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);

% --- Subplot 1: Cart Position (x) ---
subplot(4,1,1);
hold on;
y_lim_x = [-8, 0.5]; % Y-axis limits
ylim(y_lim_x);
add_control_layer(t, control_mode, T_final, y_lim_x); % Add switch line
% Add shaded region for disturbance
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], ...
      [y_lim_x(1) y_lim_x(1) y_lim_x(2) y_lim_x(2)], ...
      COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
      'DisplayName', 'Disturbance Window');
plot(t, X(1,:), 'Color', '#0090d0', 'LineWidth', 2); % Plot X
ylabel('Cart Position $x$ (m)', 'Interpreter', 'latex');
legend('Disturbance Window', 'Interpreter', 'latex');
yticks([-8:2:1]);
grid on;

% --- Subplot 2: Pendulum Angle (θ) ---
subplot(4,1,2);
hold on;
y_lim_theta = [0, 6]; % Y-axis limits
ylim(y_lim_theta);
yline(pi, 'r--', 'LineWidth', 1.5); % Target line
add_control_layer(t, control_mode, T_final, y_lim_theta);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], ...
      [y_lim_theta(1) y_lim_theta(1) y_lim_theta(2) y_lim_theta(2)], ...
      COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
plot(t, X(2,:), 'Color', '#edb120', 'LineWidth', 2, ...
     'DisplayName', 'Wrapped $\theta$'); % Plot Theta
legend('Target $\pi$','Interpreter', 'latex');
ylabel('Angle $\theta$ (rad)', 'Interpreter', 'latex');
yticks([0:2:6]);
grid on;

% --- Subplot 3: Cart Velocity (ẋ) ---
subplot(4,1,3);
hold on;
y_lim_dx = [-3, 1.5]; % Y-axis limits
ylim(y_lim_dx);
add_control_layer(t, control_mode, T_final, y_lim_dx);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], ...
      [y_lim_dx(1) y_lim_dx(1) y_lim_dx(2) y_lim_dx(2)], ...
      COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
plot(t, X(3,:), 'Color', '#dd5400', 'LineWidth', 2); % Plot dx
ylabel('Cart Vel. $\dot{x}$ (m/s)', 'Interpreter', 'latex');
yticks([-3:1.5:1.5]);
grid on;

% --- Subplot 4: Angular Velocity (θ̇) ---
subplot(4,1,4);
hold on;
y_lim_dtheta = [-15, 20]; % Y-axis limits
ylim(y_lim_dtheta);
add_control_layer(t, control_mode, T_final, y_lim_dtheta);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], ...
      [y_lim_dtheta(1) y_lim_dtheta(1) y_lim_dtheta(2) y_lim_dtheta(2)], ...
      COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
plot(t, X(4,:), 'Color', '#660099', 'LineWidth', 2); % Plot dtheta
ylabel('Ang. Vel. $\dot{\theta}$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)');
yticks([-15:5:20]);
grid on;

%% --- Figure 3: Control Force (Separated from Energy) ---
figure('Name', 'Figure 3: Control Force Evolution', 'Position', [100, 650, 800, 250]);
hold on;
y_lim_F = [-F_max*1.1, F_max*1.1]; % Use F_max for limits with margin
ylim(y_lim_F);

% Disturbance Plotting (Difference between total and controller force)
plot(t, F_history - F_controller_history, 'Color', COLOR_DIST, 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'Disturbance $F_{ext}$'); 

% The switch boundary (dashed line) is drawn here
add_control_layer(t, control_mode, T_final, y_lim_F);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], [y_lim_F(1) y_lim_F(1) y_lim_F(2) y_lim_F(2)], COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Disturbance Window');

energy_idx = (control_mode == 1);
pd_idx = (control_mode == 2);

% Plot Bang-Bang Controller Force (Yellow/Green)
plot(t(energy_idx), F_controller_history(energy_idx), 'Color', '#3baa32', 'LineWidth', 2, 'DisplayName', 'Controller Force (Bang-Bang)');
% Plot PD Controller Force (Blue)
plot(t(pd_idx), F_controller_history(pd_idx), 'Color', '#007BFF', 'LineWidth', 2, 'DisplayName', 'Controller Force (PD)');

yline(F_max, 'k--', 'LineWidth', 1, 'DisplayName', 'Controller Limit $\pm F_{max}$');
yline(-F_max, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('Force $F$ (N)', 'Interpreter', 'latex');
xlabel('Time (s)');
legend('show', 'Interpreter', 'latex', 'Location', 'best');
grid on;

%% --- Figure 5: Energy Evolution (Separated from Force) ---
figure('Name', 'Figure 5: Energy Evolution', 'Position', [100, 400, 800, 250]);
hold on;
y_lim_E = [0, max(E_history)*1.1]; % Y-axis limits
ylim(y_lim_E);
add_control_layer(t, control_mode, T_final, y_lim_E);
patch([t_disturb_start t_disturb_end t_disturb_end t_disturb_start], [y_lim_E(1) y_lim_E(1) y_lim_E(2) y_lim_E(2)], COLOR_DIST, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Disturbance Window');
plot(t, E_history, 'Color', '#3baa32', 'LineWidth', 2, ...
    'DisplayName', 'Total Energy $E(t)$');
yline(E_up, 'r--', 'LineWidth', 1.5, ...
    'DisplayName', 'Target Energy $E_{up}$');
yline(E_up + energy_tolerance, 'r:', 'LineWidth', 1, ...
    'DisplayName', 'Energy Boundary $\pm \Delta E$');
yline(E_up - energy_tolerance, 'r:', 'LineWidth', 1, ...
    'HandleVisibility', 'off'); 
xlabel('Time (s)');
ylabel('Energy (J)');
legend('show', 'Interpreter', 'latex', 'Location', 'best');
title('Pendulum Energy vs. Time');
grid on;

%% --- Figure 4: Control Mode (Position Adjusted) ---
figure('Name', 'Figure 4: Control Mode', 'Position', [900, 650, 600, 250]);
stairs(t, control_mode, 'k-', 'LineWidth', 2);
ylim([0.5, 2.5]);
yticks([1, 2]);
yticklabels({'Energy Control', 'PD Control'});
xlabel('Time (s)');
ylabel('Active Controller');
title('Control Mode Switching');
grid on;
disp('All figures generated successfully.');

%% Helper function to wrap angle to [-pi, pi]
function angle_wrapped = wrapToPi(angle)
    % Constrains an angle to the standard range (-pi, pi]
    % Used for error calculation relative to a reference
    angle_wrapped = angle - 2*pi*floor((angle + pi)/(2*pi));
end
