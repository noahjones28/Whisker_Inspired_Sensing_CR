%% Estimate applied force using simulated load
function [Fx_sim, MB_sim, MD_sim, Fx_estimate, MB_estimate, MD_estimate, F_estimate, s_estimate, z_estimate] = force_estimation_from_simulation(n, x, y, z, F_applied, s_applied, z_applied, E, I, G, J, plot_graph, Fx_error, MB_error)
    %% Elastica 3D Config
    % Compute the direction vector and magnitude
    close all;
    if plot_graph == false
        set(0, 'DefaultFigureVisible', 'off');
    end
    global x_tip; global y_tip; global z_tip;
    global direction;
    if size(x, 2) > 1 && size(y, 2) > 1 && size(z, 2) > 1
        magnitude = sqrt(x(end)^2 + y(end)^2 + z(end)^2); % Magnitude of the vector
        x_tip = x(end); y_tip = y(end); z_tip = z(end);
        direction = [x(end), y(end), z(end)] / magnitude;
    elseif size(x, 2) == 1 && size(y, 2) == 1 && size(z, 2) == 1
        magnitude = sqrt(x^2 + y^2 + z^2);
        x_tip = x; y_tip = y; z_tip = z;
        direction = [x_tip, y_tip, z_tip] / magnitude; % Normalize the tip vector
        % Generate points along the line
        t = linspace(0, magnitude, n); % Parameter for scaling along the line
        line_points = t'* direction; % Scale the direction vector 
        % Extract the x, y, and z coordinates
        x = line_points(:, 1)';
        y = line_points(:, 2)';
        z = line_points(:, 3)';

    end
    if plot_graph
        % Plot the line in 3D
        figure(1);
        plot3(x, y, z, '-o','Marker', 'none');
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        grid on;
        title('Straight Line in 3D Space');
        axis equal;
        %hold on;
    end
    % Elastica 3D
    [x3,y3,z3, F] = elastica3D(x,y,z,{[F_applied,s_applied,z_applied]},'mode','force', 'plot_steps', false, 'plot', plot_graph, 'E', E, 'I', I, 'G', G, 'J', J);
    
    % Compute the axial force (dot product of force vector and unit direction vector)
    force_vector_base = F.f;
    Fx_sim = dot(direction, force_vector_base);
    % Compute MB (moment magnitude) and MD (moment direction)
    MY_sim = F.m(2); 
    MZ_sim = F.m(3);
    MB_sim = sqrt(MY_sim^2+MZ_sim^2);
    MD_sim = atan2(MY_sim,MZ_sim);

    if isnan(Fx_sim) | isinf(Fx_sim)
        Fx_sim = 0;
    end
    if isnan(MB_sim) | isinf(MB_sim)
        MB_sim = 0;
    end
    if isnan(MD_sim) | isinf(MD_sim)
        MD_sim = 0;
    end

    % if the user defines an Fx_error or MB_error (simulation uncertianty)
    if (Fx_error ~= 0)
        Fx_sim = Fx_sim + Fx_error; % then apply error
    end
    if (MB_error ~= 0)
        MB_sim = MB_sim + MB_error; % then apply error
    end

    current_fig = gcf;
    if plot_graph
        saveas(current_fig, 'plot1.png');
    end
    view([0, 0]);
    % Get current Z limits
    zLimits = zlim;
    % Add padding to the Z-axis
    padding = 0.03;  % Padding amount (you can adjust this value)
    zlim([zLimits(1) - padding, zLimits(2) + padding]);
    % Get current Z limits
    xLimits = xlim;
    % Add padding to the Z-axis
    padding = 0.05;  % Padding amount (you can adjust this value)
    xlim([xLimits(1) - padding, xLimits(2) + padding]);
    % Save the plot as a PNG file
    if plot_graph
        saveas(current_fig, 'plot2.png')
    end
    % use force estimation script to calculate remaining values
    [Fx_estimate, MB_estimate, MD_estimate, F_estimate, s_estimate, z_estimate] = force_estimation_script(n, x, y, z, E, I, G, J, Fx_sim, MB_sim, MD_sim, plot_graph);
end