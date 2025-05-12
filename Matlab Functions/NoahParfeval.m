function results = NoahParfeval(n, x_dis, y_dis, z_dis, z_applied, E, I_robust, G, J_dis, s_values, F_values)
    % Create a parallel pool if one isn't already open
    if isempty(gcp('nocreate'))
        % Get the local cluster object
        localCluster = parcluster('Noah12');
        parpool(localCluster)
    end

    % Initialize an empty array for futures
    futures(1:length(s_values)) = parallel.FevalFuture;
    % Loop through all sampled points
    for index = 1:length(s_values)  % Get the index directly
        s = s_values(index);
        F = F_values(index);    
        % Launch Noah_Force_Matlab asynchronously
        future = parfeval(@Noah_Force_Matlab, 9, n, x_dis, y_dis, z_dis, F, s, z_applied, E, I_robust, G, J_dis);
        futures(index) = future;
    end
    % Initialize an array to store futures
    results = zeros(9, length(futures));
    % Fetch outputs for all futures
    for i = 1:length(futures)
        [Fx_model, MB_model, MD_model, Fx_estimate, MB_estimate, MD_estimate, F_opt, s_opt, z_opt] = fetchOutputs(futures(i));
        % Store outputs as a column in the matrix
        results(:, i) = [Fx_model, MB_model, MD_model, Fx_estimate, MB_estimate, MD_estimate, F_opt, s_opt, z_opt];
    end
end
