%% Force Estimation Script
function [Fx_estimate, MB_estimate, MD_estimate, F_estimate, s_estimate, z_estimate] = force_estimation_script(n, x, y, z, E, I, G, J, Fx, MB, MD) 
    % Define sensed values
    proximal_values_model = [Fx, MB, MD];
    % Define initial guess for [F, s, z]
    magnitude = sqrt(x(end)^2 + y(end)^2 + z(end)^2); % Magnitude of the vector
    direction = [x(end), y(end), z(end)] / magnitude;
    params0 = [0.5, magnitude/2, 0]; % Replace with reasonable guesses
    % Set options for fmincon with the custom output function
    % Pass additional arguments using an anonymous function
    outputFcnWithArgs = @(params, optimValues, state) ...
        myOutputFcn(params, optimValues, state, proximal_values_model, x, y, z, E, I, G, J);
    if display 
        options = optimoptions('fmincon', 'OutputFcn', outputFcnWithArgs, 'Display', 'iter', 'StepTolerance', 1e-20,'MaxIterations', 300);
    else
        options = optimoptions('fmincon', 'OutputFcn', outputFcnWithArgs, 'Display', 'none', 'StepTolerance', 1e-20,'MaxIterations', 300);
    end
    % Define lower and upper bounds for each variable
    F_min = 0; F_max = 1;  % Example bounds for F
    s_min = 0; s_max = magnitude;   % Example bounds for s
    z_min = 0; z_max = 2*pi; % Example bounds for z
    % Combine into bound vectors
    lb = [F_min, s_min, z_min]; % Lower bounds
    ub = [F_max, s_max, z_max]; % Upper bounds
    % Define weights
    weights = [1, 10, 1];  % Reduce the influence of the third residual
    % Call fmincon
    [optimized_params, fval] = fmincon(@(params) sum((weights.*residual_function(params, proximal_values_model, x, y, z, E, I, G, J)).^2), params0, [], [], [], [], lb, ub, [], options);
    % Extract optimized F, s, z
    F_opt = optimized_params(1);
    s_opt = optimized_params(2);
    z_opt = optimized_params(3);
    
    % Check solution
    % Elastica 3D
    [x3,y3,z3, F] = elastica3D(x,y,z,{[F_opt,s_opt,z_opt]},'mode','force', 'plot_steps', false, 'plot', plot_graph, 'E', E, 'I', I, 'G', G, 'J', J);
    % Compute the axial force (dot product of force vector and unit direction vector)
    force_vector_base = F.f;
    Fx_estimate = dot(direction, force_vector_base);
    % Compute MB (moment magnitude) and MD (moment direction)
    MY_estimate = F.m(2); 
    MZ_estimate = F.m(3);
    MB_estimate = sqrt(MY_estimate^2+MZ_estimate^2);
    MD_estimate = atan2(MY_estimate,MZ_estimate);
    
    if plot_graph
        saveas(gcf, 'plot3.png')
    end
    
    % Residual Function
    function res = residual_function(params, proximal_values_model, x, y, z, E, I, G, J)
        magnitude = sqrt(x(end)^2 + y(end)^2 + z(end)^2); % Magnitude of the vector
        direction = [x(end), y(end), z(end)] / magnitude;
        % Unpack parameters
        F = params(1);
        s = params(2);
        z_input = params(3);
        % Call the elastica3D algorithm
        [~, ~, ~, F_struct] = elastica3D(x, y, z, {[F, s, z_input]}, 'mode', 'force', 'plot_steps', false, 'plot', false, 'E', E, 'I', I, 'G', G, 'J', J);
        % Extract estimated a, b, c
        % Compute the axial force (dot product of force vector and unit direction vector)
        force_vector_base = F_struct.f;
        Fx = dot(direction, force_vector_base);
        % Compute MB (moment magnitude) and MD (moment direction)
        MY = F_struct.m(2);
        MZ = F_struct.m(3);
        MB = sqrt(MY^2+MZ^2);
        MD = atan2(MY,MZ);
        if isnan(Fx) | isinf(Fx)
            Fx = 0;
        end
        if isnan(MB) | isinf(MB)
            MB = 0;
        end
        if isnan(MD) | isinf(MD)
            MD = 0;
        end
        estimated_abc = [Fx, MB, MD];
        % Compute residuals
        %fprintf('Fx:%.6f, MBx:%.6f, MD:%.6f\n', Fx, MB, MD);
        res = estimated_abc - proximal_values_model;
    end
    
    % Create the custom output function
    function stop = myOutputFcn(params, optimValues, state, proximal_values_model, x, y, z, E, I, G, J)
        global display;
        % Access the current parameters and calculate residuals
        residuals = residual_function(params, proximal_values_model, x, y, z, E, I, G, J);
        sum_of_residuals = sum(residuals.^2);
        
        if display
            % Print the parameters and the sum of residuals
            fprintf('Iteration %d: F = %.6f, s = %.6f, z = %.6f, Sum of residuals = %.6f\n', ...
                optimValues.iteration, params(1), params(2), params(3), sum_of_residuals);
        end
        % Stop condition (set to false to continue optimization)
        stop = false;
    end
end






